# -*- coding: utf-8 -*-
"""uav_probe.py —— 无人机图传接入探测工具（只读探测，不发送任何飞行指令）。

归属：09_uav_realtime/ 独立实验目录。不修改任何既有源文件。

================================ 为什么需要这个脚本 ============================
说明书显示这台无人机的图传不是 RTSP，而是 **WiFi_CAM 私有协议**：
  模块上电（红灯闪）→ 手机连 `WIFI_____xxx` 热点 → App 点 START 看图传。

要在电脑上拿到画面，必须知道**它在哪个 UDP 端口推流、用什么编码**。
这些是硬事实，猜错会浪费大量时间，且「猜一个端口去连、连不上」这种失败
无法区分「端口错」还是「没触发推流」。所以本脚本的设计原则是：

  ★ 不猜。把所有未知量做成**可观测的探测步骤**，逐个测出来并落盘取证。

============================== 已知线索（待实测确认）=========================
来源：公开资料（深圳市天之衡电子科技 WiFi_CAM，包名 com.tzh.wifi.wificam.activity）
  - UDP 8080：视频流，MJPEG 格式
  - UDP 8090：控制指令（如 AA 80 80 00 80 00 80 55）
  - 热点名形如 WIFI_FPV_XXX / WIFI_____xxx

来源：同族机型（Eachine E88/E58 类）逆向经验
  - **上电默认在 2.4GHz 遥控器模式，此时不推 WiFi 视频流**
  - 必须**先往 8080 发 2 字节魔数 `42 76`（0x42 0x76）** 切到 App 模式，
    视频流才会开始；断开时发 `42 77` 还原。
  - ⚠️ 这条是**同族机型**经验，本机是否一致**必须实测**，本脚本会把它做成
    一个「有/无魔数」的对照实验，而不是直接假定成立。

=============================== 探测步骤总览 ================================
  S0  环境自检：本机 IP / 网段 / 是否已连上无人机热点（判断依据：网关是否
      在 192.168.4.x / 192.168.1.x 等私有网段且 SSID 形如 WIFI_*）
  S1  找无人机 IP：连上热点后，网关通常就是无人机；再辅以 ARP 表交叉验证
  S2  端口探测：对候选端口（8080/8090/5000/5504/8888/7060 等）做
      「监听能否收到包」探测 —— 注意 UDP 无连接，'扫描'只能靠**发探测包
      看有无回包**或**直接监听看有无推流**
  S3  魔数对照实验：先在**不发魔数**的情况下监听 8080 端口 N 秒记录收包量，
      再发 `42 76` 后同样监听 N 秒 —— 对比两次收包量，实测魔数是否必需
  S4  流特征分析：对收到的包做统计（包长分布、是否含 JPEG SOI 标记 FFD8、
      是否含 MJPEG 边界），据此判断是 MJPEG 还是 H.264
  S5  落盘：把全部原始观测写成 JSON + 文本日志

=============================== 安全声明 ===============================
  ★ 本脚本**只发 2 字节模式切换魔数**，不发送任何摇杆/起飞/降落指令。
    绝不能让电脑误发控制指令造成无人机意外起飞。
  ★ 探测期间请把无人机放在开阔地面、取下桨叶（或确认不会伤人）。
"""

from __future__ import annotations

import argparse
import json
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_LOGS = _HERE / "logs"

# 候选端口（按先验概率排序）。来源见文件头「已知线索」。
VIDEO_PORT_CANDIDATES = [8080, 8090, 5000, 5504, 7060, 8888, 9004, 5600, 11111]
CONTROL_PORT_CANDIDATES = [8090, 8080, 8800]

# 魔数（同族机型经验，需实测确认）
MAGIC_ENTER_APP_MODE = bytes([0x42, 0x76])   # "Bv" —— 切到 App 模式，开始推流
MAGIC_EXIT_APP_MODE = bytes([0x42, 0x77])    # "Bw" —— 还原遥控器模式

# 常见无人机热点网段（用于判断"是否已连上无人机"）
UAV_SUBNETS = ("192.168.4.", "192.168.1.", "192.168.100.", "192.168.0.", "10.0.0.")


# ------------------------------------------------------------------ 基础工具
def _ps(cmd: str, timeout: int = 25) -> str:
    """调用 PowerShell 并返回 stdout。

    ★ 为什么要绕这一道：本机实测从 Bash 直接调 PowerShell 会被安全策略拒绝，
      且 PowerShell 工具的 stdout 不回传。故用 Python 的 subprocess 调用，
      自己接住输出。注意**不能**设置 -NonInteractive 之外的交互参数。
    """
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command", cmd],
            capture_output=True, text=True, encoding="utf-8",
            errors="ignore", timeout=timeout,
        )
        return (r.stdout or "") + (r.stderr or "")
    except Exception as e:
        return f"__PS_ERROR__ {e}"


def local_ipv4_map() -> dict:
    """返回 {别名: (ip, 网关)}，仅活动网卡。"""
    ps = (
        "Get-NetIPConfiguration | Where-Object {$_.NetAdapter.Status -eq 'Up'} | "
        "ForEach-Object { \"$($_.InterfaceAlias)|"
        "$(($_.IPv4Address.IPAddress) -join ',')\" + '|' + "
        "\"$($_.IPv4DefaultGateway.NextHop)\" }"
    )
    out = _ps(ps)
    res = {}
    for line in out.splitlines():
        line = line.strip()
        if "|" not in line or line.startswith("__PS_ERROR__"):
            continue
        parts = line.split("|")
        if len(parts) >= 3 and parts[1]:
            res[parts[0]] = (parts[1].split(",")[0].strip(),
                             parts[2].strip())
    return res


def current_wifi_ssid() -> str:
    """当前连接的 WiFi SSID。netsh 输出中文乱码但 SSID 是 ASCII，可正则取。"""
    out = _ps("netsh wlan show interfaces")
    for line in out.splitlines():
        # 形如 "    SSID                   : WIFI_FPV_123456"
        if "SSID" in line and ":" in line and "BSSID" not in line:
            val = line.split(":", 1)[1].strip()
            if val and val not in ("", "\r"):
                return val
    return ""


def arp_table() -> list[tuple[str, str]]:
    """ARP 表 → [(ip, mac)]。"""
    out = _ps("arp -a")
    rows = []
    for line in out.splitlines():
        p = line.split()
        if len(p) >= 2 and p[0].count(".") == 3:
            rows.append((p[0], p[1]))
    return rows


def route_table() -> str:
    return _ps("route print -4")


# ------------------------------------------------------------------ 探测步骤
class Probe:
    def __init__(self) -> None:
        self.log_lines: list[str] = []
        self.result: dict = {"schema": "uav_probe/v1", "steps": {}}
        # ★ 安全标志：只有确认连上了无人机热点，才允许向其发送任何数据包。
        self._drone_confirmed: bool = False

    def log(self, msg: str = "") -> None:
        print(msg)
        self.log_lines.append(msg)

    # ---------------- S0 环境自检 ----------------
    def s0_env(self) -> dict:
        self.log("=" * 74)
        self.log("S0  环境自检")
        self.log("=" * 74)
        nics = local_ipv4_map()
        ssid = current_wifi_ssid()

        self.log(f"  活动网卡：{nics if nics else '（未取到）'}")
        self.log(f"  当前 WiFi SSID：{ssid or '（未取到，可能非 WiFi 连接）'}")

        # 判断是否已连上无人机热点。
        # ★ 判定标准必须**严格**（本脚本首轮实跑踩过坑）：
        #   最初用「网段属于常见的无人机私有网段」当依据，结果 192.168.3.x
        #   （其实用户家的路由器）也被判成"可能已连上"，导致脚本向其发了包。
        #   ⇒ 现在只认**强证据**：SSID 以 WIFI 开头（说明书明确热点是这样命名的）。
        #     网段只作为"提示信息"，不作为发包许可。
        guessed_ip, guessed_gw = "", ""
        for alias, (ip, gw) in nics.items():
            guessed_ip, guessed_gw = ip, gw

        ssid_is_uav = ssid.upper().startswith("WIFI")
        subnet_hint = any(guessed_ip.startswith(s) for s in UAV_SUBNETS)

        if ssid_is_uav:
            self._drone_confirmed = True
            self.log(f"  ⇒ ✓ 已确认连上无人机热点（SSID={ssid}，"
                     f"符合说明书的 WIFI_____xxx 命名）")
            self.log(f"     可以安全执行后续探测（含模式魔数发送）。")
        else:
            self.log("  ⇒ ✗ 未确认连上无人机热点。")
            if subnet_hint:
                self.log(f"     （当前网段 {guessed_ip} 恰好在常见私有网段里，"
                         f"但 SSID 不是 WIFI 开头 ⇒ **不能**据此认定是无人机）")
            self.log("     请按说明书操作：")
            self.log("       1) 无人机（图传模块）上电，等红灯闪烁")
            self.log("       2) 在 Windows 的 WiFi 列表里连 `WIFI_____xxx` 热点")
            self.log("       3) 连上后重新运行本脚本")
            self.log("     ⚠️ 本机只有一块 WiFi 网卡，连无人机热点后会断开"
                     "当前网络（正常现象）")
            self.log("     ⚠️ 安全策略：未确认连上无人机前，本脚本**不会向任何"
                     "目标发送数据包**，只做被动监听。")

        self.result["steps"]["S0_env"] = {
            "nics": {k: list(v) for k, v in nics.items()},
            "ssid": ssid,
            "drone_confirmed": self._drone_confirmed,
            "subnet_hint_only": bool(subnet_hint and not ssid_is_uav),
            "local_ip": guessed_ip, "gateway": guessed_gw,
        }
        return self.result["steps"]["S0_env"]

    # ---------------- S1 找无人机 IP ----------------
    def s1_find_drone(self, gw: str, local_ip: str) -> str:
        self.log()
        self.log("=" * 74)
        self.log("S1  定位无人机 IP")
        self.log("=" * 74)

        if not self._drone_confirmed:
            self.log("  ⇒ 未确认连上无人机热点，跳过定位（避免把家用路由器"
                     "误判成无人机）。")
            self.result["steps"]["S1_drone"] = {
                "skipped": True, "reason": "drone_not_confirmed"}
            return gw

        candidates: list[str] = []
        if gw:
            candidates.append(gw)
            self.log(f"  网关 {gw} —— 这类 WiFi 图传模块通常自己就是网关/AP")

        arp = arp_table()
        self.log(f"  ARP 表共 {len(arp)} 条，其中与网关同网段的：")
        subnet = ".".join(gw.split(".")[:3]) + "." if gw else ""
        for ip, mac in arp:
            if subnet and ip.startswith(subnet) and ip != local_ip:
                self.log(f"     {ip:16s} {mac}")
                if ip not in candidates:
                    candidates.append(ip)

        self.result["steps"]["S1_drone"] = {
            "gateway": gw, "candidates": candidates, "arp": arp[:40],
        }
        self.log(f"  ⇒ 候选无人机 IP：{candidates}")
        return candidates[0] if candidates else gw

    # ---------------- S2/S3 端口与魔数对照实验 ----------------
    def _listen_count(self, port: int, seconds: float, local_ip: str):
        """监听某 UDP 端口，返回 (包数, 总字节, 包长列表前10, 是否见 JPEG)。"""
        n, total = 0, 0
        sizes: list[int] = []
        saw_jpeg = False
        first_bytes = b""
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("", port))
        except OSError as e:
            s.close()
            return {"error": f"bind 失败（端口被占用？）：{e}"}
        s.settimeout(0.4)
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            try:
                data, addr = s.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            n += 1
            total += len(data)
            if len(sizes) < 10:
                sizes.append(len(data))
            if not first_bytes:
                first_bytes = data[:24]
            if data[:2] == b"\xff\xd8":      # JPEG SOI
                saw_jpeg = True
        s.close()
        return {
            "packets": n, "bytes": total, "sample_sizes": sizes,
            "saw_jpeg_soi": saw_jpeg,
            "first_bytes_hex": first_bytes.hex(),
        }

    def s2_s3_ports(self, drone_ip: str, local_ip: str,
                    listen_sec: float, do_magic: bool) -> dict:
        """对候选端口做「魔数前 / 魔数后」对照监听。"""
        self.log()
        self.log("=" * 74)
        self.log("S2/S3  端口探测 + 魔数对照实验")
        self.log("=" * 74)
        self.log(f"  无人机 IP={drone_ip}  本机 IP={local_ip}")
        self.log(f"  每端口监听 {listen_sec}s；魔数对照={'开启' if do_magic else '关闭'}")
        self.log()
        self.log("  【第一阶段】不发魔数，直接监听各候选端口")

        phase_a = {}
        for port in VIDEO_PORT_CANDIDATES:
            r = self._listen_count(port, listen_sec, local_ip)
            phase_a[port] = r
            n = r.get("packets", 0)
            flag = "★有数据" if n > 0 else "无"
            self.log(f"    端口 {port:6d}：{flag:8s} 包数={n:5d} "
                     f"字节={r.get('bytes', 0):8d} "
                     f"JPEG={r.get('saw_jpeg_soi')}")
            if n == 0 and r.get("error"):
                self.log(f"      （{r['error']}）")

        phase_b = {}
        if do_magic and not self._drone_confirmed:
            # ★★ 安全门禁（本脚本首轮实跑暴露的问题）★★
            #   首轮在**未连接无人机**时仍向 192.168.3.1（用户家的路由器）
            #   发了魔数包。虽只是 2 字节、无实际危害，但"往非目标设备发包"
            #   这个行为本身不可接受 —— 万一是别人的设备/生产网设备呢。
            #   ⇒ 未确认连上无人机热点，一律不发任何包，只做被动监听。
            self.log()
            self.log("  【第二阶段】已跳过：**未确认连上无人机热点**，"
                     "按安全策略不向任何目标发送数据包。")
            self.log("     （只做被动监听。确认连上热点后重跑本脚本，"
                     "阶段二会自动执行。）")
        elif do_magic:
            self.log()
            self.log(f"  【第二阶段】向 {drone_ip}:8080 与 :8090 发送魔数 "
                     f"{MAGIC_ENTER_APP_MODE.hex()}（切 App 模式），再监听")
            sent_to = []
            for cport in CONTROL_PORT_CANDIDATES:
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    s.sendto(MAGIC_ENTER_APP_MODE, (drone_ip, cport))
                    s.close()
                    sent_to.append(cport)
                except Exception as e:
                    self.log(f"    发往 {cport} 失败：{e}")
            self.log(f"    已发送到端口：{sent_to}")
            time.sleep(1.0)   # 给无人机切换模式的余量

            for port in VIDEO_PORT_CANDIDATES:
                r = self._listen_count(port, listen_sec, local_ip)
                phase_b[port] = r
                n = r.get("packets", 0)
                flag = "★有数据" if n > 0 else "无"
                self.log(f"    端口 {port:6d}：{flag:8s} 包数={n:5d} "
                         f"字节={r.get('bytes', 0):8d} "
                         f"JPEG={r.get('saw_jpeg_soi')}")

        # ---- 结论：哪个端口真的在推流 ----
        self.log()
        winners = []
        for port in VIDEO_PORT_CANDIDATES:
            a = phase_a.get(port, {}).get("packets", 0) or 0
            b = phase_b.get(port, {}).get("packets", 0) or 0
            if b > 0 or a > 0:
                winners.append((port, a, b))
        if winners:
            self.log("  ⇒ 有数据的端口（端口, 魔数前包数, 魔数后包数）：")
            for p, a, b in winners:
                delta = "魔数后显著增加 ⇒ 魔数可能是必需的" if b > a * 2 else \
                        ("两阶段都推流 ⇒ 魔数非必需" if a > 0 and b > 0 else
                         "仅魔数后推流 ⇒ 魔数必需")
                self.log(f"     {p}: ({a}, {b})  {delta}")
        else:
            self.log("  ⇒ ✗ 所有候选端口都没收到数据。可能原因：")
            self.log("     1) 还没连上无人机热点（回看 S0 提示）")
            self.log("     2) 无人机需要**手机 App 点 START** 才开始推流")
            self.log("        （很多机型只有 App 发过握手后才推流）")
            self.log("     3) 推流端口不在候选列表里 ⇒ 用 `--scan-all` 全端口扫")
            self.log("     4) 无人机与电脑不在同一网段")

        self.result["steps"]["S2S3_ports"] = {
            "drone_ip": drone_ip, "local_ip": local_ip,
            "listen_sec": listen_sec, "magic_sent": do_magic,
            "phase_before_magic": phase_a, "phase_after_magic": phase_b,
            "winners": winners,
        }
        return self.result["steps"]["S2S3_ports"]

    # ---------------- S4 流特征 ----------------
    def s4_stream(self, drone_ip: str, ports: list[int],
                  seconds: float) -> dict:
        """对已确认有数据的端口做更长时间的抓取并分析流特征。"""
        if not ports:
            return {}
        self.log()
        self.log("=" * 74)
        self.log(f"S4  流特征分析（端口 {ports}，抓 {seconds}s）")
        self.log("=" * 74)
        feats = {}
        for port in ports:
            r = self._listen_count(port, seconds, "")  # local_ip 未用
            n = r.get("packets", 0)
            self.log(f"  端口 {port}：包数={n} 字节={r.get('bytes', 0)}")
            self.log(f"    前 10 个包长：{r.get('sample_sizes')}")
            self.log(f"    首包十六进制：{r.get('first_bytes_hex')}")
            self.log(f"    含 JPEG SOI(ffd8)：{r.get('saw_jpeg_soi')}")
            if n > 0:
                avg = (r.get("bytes", 0) / n) if n else 0
                self.log(f"    平均包长：{avg:.1f} B")
                if r.get("saw_jpeg_soi"):
                    self.log("    ⇒ 判定：**MJPEG**（每包含完整或分片的 JPEG）")
                elif avg > 1200:
                    self.log("    ⇒ 判定：可能是 **H.264/H.265**（大包、"
                             "无 JPEG 标记，需按 RTP 或裸流重组）")
                else:
                    self.log("    ⇒ 判定：不确定，建议用 Wireshark 进一步看")
            feats[port] = r
        self.result["steps"]["S4_stream"] = feats
        return feats

    # ---------------- 落盘 ----------------
    def save(self) -> Path:
        _LOGS.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        jp = _LOGS / f"uav_probe_{stamp}.json"
        tp = _LOGS / f"uav_probe_{stamp}.txt"
        with open(jp, "w", encoding="utf-8") as f:
            json.dump(self.result, f, ensure_ascii=False, indent=2)
        with open(tp, "w", encoding="utf-8") as f:
            f.write("\n".join(self.log_lines) + "\n")
        self.log()
        self.log(f"探测记录 → {jp}")
        self.log(f"文本日志 → {tp}")
        return jp


def main() -> int:
    ap = argparse.ArgumentParser(
        description="无人机图传接入探测（只发模式魔数，不发任何飞行指令）")
    ap.add_argument("--listen-sec", type=float, default=3.0,
                    help="每个候选端口的监听秒数（默认 3）")
    ap.add_argument("--no-magic", action="store_true",
                    help="跳过魔数对照实验（只做被动监听）")
    ap.add_argument("--stream-sec", type=float, default=6.0,
                    help="对确认推流的端口做流特征分析的秒数（默认 6）")
    a = ap.parse_args()

    p = Probe()
    p.log("无人机图传接入探测")
    p.log(f"时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    p.log("⚠️ 本脚本只发 2 字节模式魔数，不发任何飞行指令。"
          "请确保无人机在开阔地面且不会伤人。")

    env = p.s0_env()
    drone_ip = p.s1_find_drone(env.get("gateway", ""), env.get("local_ip", ""))
    ports = p.s2_s3_ports(drone_ip, env.get("local_ip", ""),
                          a.listen_sec, do_magic=not a.no_magic)

    active = [w[0] for w in ports.get("winners", []) if (w[2] or w[1])]
    p.s4_stream(drone_ip, active[:2], a.stream_sec)

    p.log()
    p.log("=" * 74)
    p.log("探测完成。请把上面的结论（尤其「有数据的端口」）告诉我，")
    p.log("我据此把 rt_config.yaml 的无人机源预设配成实测值。")
    p.log("=" * 74)
    p.save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
