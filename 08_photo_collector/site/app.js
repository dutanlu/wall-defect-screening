/* ==========================================================================
 * 页面逻辑：登录 / 注册 → 选照片 + 填拍摄距离 → 上传 → 写入记录
 * ========================================================================== */
(function () {
  'use strict';

  const cfg = window.APP_PUBLIC_CONFIG;
  const L = window.APP_LIMITS;
  const $ = (id) => document.getElementById(id);

  let pickedFile = null;
  let pickedDims = { w: null, h: null };

  /* ------------------------------------------------------------------ 通用 */
  function setMsg(el, text, kind) {
    el.textContent = text || '';
    el.className = 'msg' + (kind ? ' msg--' + kind : '');
    el.hidden = !text;
  }
  function busy(btn, on, label) {
    btn.disabled = on;
    if (on) {
      btn.dataset.label = btn.textContent;
      btn.textContent = label || '处理中…';
    } else if (btn.dataset.label) {
      btn.textContent = btn.dataset.label;
    }
  }
  function fmtSize(bytes) {
    if (!bytes && bytes !== 0) return '—';
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(0) + ' KB';
    return (bytes / 1024 / 1024).toFixed(1) + ' MB';
  }

  /* ------------------------------------------------------------ 视图切换 */
  function showView(name) {
    $('view-auth').hidden = name !== 'auth';
    $('view-main').hidden = name !== 'main';
  }
  function renderUser(session) {
    const signed = !!(session && session.user && session.user.email);
    $('who').textContent = signed ? session.user.email : '';
    const dot = $('who-dot');
    if (dot) dot.hidden = !signed;
    const out = $('signout-btn');
    if (out) out.hidden = !signed;
  }

  /* --------------------------------------------------------------- 登录页 */
  let authMode = 'signin'; // signin | signup
  function setAuthMode(mode) {
    authMode = mode;
    document.querySelectorAll('.tab').forEach((t) => {
      t.classList.toggle('is-active', t.dataset.mode === mode);
    });
    $('auth-submit').textContent = mode === 'signup' ? '注册并登录' : '登录';
    $('auth-extra').hidden = mode !== 'signup';
    setMsg($('auth-msg'), '');
  }

  document.querySelectorAll('.tab').forEach((t) => {
    t.addEventListener('click', () => setAuthMode(t.dataset.mode));
  });

  // 邮箱验证码登录（OTP）：与密码登录并列的另外一种方式
  $('otp-btn').addEventListener('click', async () => {
    const cloud = window.__CLOUD_READY__;
    const email = $('auth-email').value.trim();
    if (!email) return setMsg($('auth-msg'), '请先填写邮箱。', 'err');
    busy($('otp-btn'), true, '发送中…');
    setMsg($('auth-msg'), '');
    try {
      const started = await cloud.auth.signInWithOtp({ email });
      if (started.error) throw started.error;
      window.__otpPending = started.data;
      $('otp-row').hidden = false;
      setMsg($('auth-msg'), '验证码已发送，请查收邮箱（含垃圾箱）。', 'ok');
    } catch (e) {
      setMsg($('auth-msg'), (e && e.message) || '验证码发送失败，请稍后再试。', 'err');
    } finally {
      busy($('otp-btn'), false);
    }
  });

  $('otp-verify-btn').addEventListener('click', async () => {
    const code = $('otp-code').value.trim();
    if (!code) return setMsg($('auth-msg'), '请填写验证码。', 'err');
    if (!window.__otpPending) return setMsg($('auth-msg'), '请先点击"获取验证码"。', 'err');
    busy($('otp-verify-btn'), true, '验证中…');
    try {
      const done = await window.__otpPending.verify({ token: code });
      if (done.error) throw done.error;
      window.__otpPending = null;
      await afterSignIn();
    } catch (e) {
      setMsg($('auth-msg'), (e && e.message) || '验证码不正确或已过期。', 'err');
    } finally {
      busy($('otp-verify-btn'), false);
    }
  });

  // 密码登录 / 带密码的注册
  $('auth-form').addEventListener('submit', async (ev) => {
    ev.preventDefault();
    const cloud = window.__CLOUD_READY__;
    const email = $('auth-email').value.trim();
    const password = $('auth-password').value;
    if (!email || !password) return setMsg($('auth-msg'), '请填写邮箱与密码。', 'err');

    busy($('auth-submit'), true);
    setMsg($('auth-msg'), '');
    try {
      if (authMode === 'signin') {
        const { error } = await cloud.auth.signInWithPassword({ email, password });
        if (error) throw error;
        await afterSignIn();
      } else {
        // 注册必须先验证邮箱，再把密码带上
        const sent = await cloud.auth.sendOtp({ email });
        if (sent.error) throw sent.error;
        window.__signupPending = { email, password, sent: sent.data };
        $('signup-otp-row').hidden = false;
        setMsg($('auth-msg'), '验证码已发送到邮箱，填写后可完成注册。', 'ok');
      }
    } catch (e) {
      setMsg($('auth-msg'), '账号或密码不正确，请重试。', 'err');
    } finally {
      busy($('auth-submit'), false);
    }
  });

  $('signup-verify-btn').addEventListener('click', async () => {
    const cloud = window.__CLOUD_READY__;
    const code = $('signup-code').value.trim();
    const pending = window.__signupPending;
    if (!code || !pending) return setMsg($('auth-msg'), '请先填写邮箱并获取验证码。', 'err');
    busy($('signup-verify-btn'), true, '提交中…');
    try {
      const completed = await cloud.auth.verifyOtp({
        verificationId: pending.sent.verificationId,
        token: code,
        email: pending.email,
        isExistingUser: pending.sent.isExistingUser,
        password: pending.sent.isExistingUser ? undefined : pending.password,
      });
      if (completed.error) throw completed.error;
      window.__signupPending = null;
      await afterSignIn();
    } catch (e) {
      setMsg($('auth-msg'), (e && e.message) || '验证失败，请检查验证码。', 'err');
    } finally {
      busy($('signup-verify-btn'), false);
    }
  });

  $('signout-btn').addEventListener('click', async () => {
    const cloud = window.__CLOUD_READY__;
    await cloud.auth.signOut();
    showView('auth');
  });

  /* --------------------------------------------------------------- 主页面 */
  $('pick-btn').addEventListener('click', () => $('file-input').click());

  $('file-input').addEventListener('change', (ev) => {
    const f = ev.target.files && ev.target.files[0];
    setMsg($('form-msg'), '');
    if (!f) return;
    if (!L.allowedTypes.includes(f.type)) {
      pickedFile = null;
      $('file-meta').hidden = true;
      $('upload-btn').disabled = true;
      return setMsg($('form-msg'), '只接受 JPG / PNG / WebP 格式的图片。', 'err');
    }
    if (f.size > L.maxFileBytes) {
      pickedFile = null;
      $('file-meta').hidden = true;
      $('upload-btn').disabled = true;
      return setMsg($('form-msg'), `图片 ${fmtSize(f.size)} 超过 ${fmtSize(L.maxFileBytes)} 上限，请压缩后再传。`, 'err');
    }
    pickedFile = f;
    pickedDims = { w: null, h: null };
    const img = new Image();
    img.onload = () => {
      pickedDims = { w: img.naturalWidth, h: img.naturalHeight };
      renderPicker(f);
    };
    img.onerror = () => renderPicker(f);
    img.src = URL.createObjectURL(f);
  });

  function renderPicker(f) {
    $('file-meta').hidden = false;
    $('file-meta').innerHTML =
      '<div class="thumb"><img src="' + URL.createObjectURL(f) + '" alt="预览"></div>' +
      '<div class="meta">' +
      '<div class="meta__name">' + escapeHtml(f.name) + '</div>' +
      '<div class="meta__sub">' + fmtSize(f.size) +
      (pickedDims.w ? ' · ' + pickedDims.w + '×' + pickedDims.h : '') + '</div>' +
      '</div>';
    $('upload-btn').disabled = false;
  }

  // 距离快选
  document.querySelectorAll('.chip[data-dist]').forEach((c) => {
    c.addEventListener('click', () => {
      $('distance').value = c.dataset.dist;
      $('distance-hint').value = c.dataset.hint || '';
      document.querySelectorAll('.chip[data-dist]').forEach((x) => x.classList.remove('is-active'));
      c.classList.add('is-active');
    });
  });

  $('upload-btn').addEventListener('click', async () => {
    const store = window.__CloudStore;
    if (!store) return setMsg($('form-msg'), '云服务尚未就绪，请刷新页面重试。', 'err');

    const distRaw = $('distance').value.trim();
    const dist = Number(distRaw);
    if (!distRaw || !isFinite(dist)) {
      return setMsg($('form-msg'), '请填写拍摄距离（米）。', 'err');
    }
    if (dist < L.minDistance || dist > L.maxDistance) {
      return setMsg($('form-msg'), `拍摄距离需在 ${L.minDistance}–${L.maxDistance} 米之间。`, 'err');
    }
    if (!pickedFile) return setMsg($('form-msg'), '请先选择一张照片。', 'err');

    const { session, error } = await store.getSession();
    if (error || !session) {
      showView('auth');
      return setMsg($('auth-msg'), '登录状态已失效，请重新登录。', 'err');
    }

    busy($('upload-btn'), true, '上传中…');
    setMsg($('form-msg'), '');

    let photoKey = null;
    try {
      photoKey = await store.uploadPhoto(session, pickedFile);
    } catch (e) {
      busy($('upload-btn'), false);
      return setMsg($('form-msg'), '照片上传失败：' + ((e && e.message) || '网络异常，请重试。'), 'err');
    }

    try {
      await store.insertSubmission({
        contributor_name: $('contributor').value.trim() || null,
        building_note: $('note').value.trim() || null,
        shot_distance_m: dist,
        distance_hint: $('distance-hint').value.trim() || null,
        photo_key: photoKey,
        photo_name: pickedFile.name,
        photo_size: pickedFile.size,
        photo_type: pickedFile.type,
        image_width: pickedDims.w,
        image_height: pickedDims.h,
      });
    } catch (e) {
      busy($('upload-btn'), false);
      return setMsg($('form-msg'),
        '照片已上传，但记录保存失败：' + ((e && e.message) || '请稍后重试。'), 'err');
    }

    busy($('upload-btn'), false);
    setMsg($('form-msg'), '提交成功，感谢你的帮忙！', 'ok');
    resetForm();
    await refreshStats();
  });

  function resetForm() {
    pickedFile = null;
    pickedDims = { w: null, h: null };
    $('file-input').value = '';
    $('file-meta').hidden = true;
    $('file-meta').innerHTML = '';
    $('upload-btn').disabled = true;
    $('distance').value = '';
    $('distance-hint').value = '';
    $('note').value = '';
    document.querySelectorAll('.chip[data-dist]').forEach((x) => x.classList.remove('is-active'));
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) =>
      ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }

  /* ----------------------------------------------------------- 统计与列表 */
  async function refreshStats() {
    const store = window.__CloudStore;
    if (!store) return;
    try {
      const [n, rows] = await Promise.all([store.countSubmissions(), store.listSubmissions(12)]);
      $('stat-count').textContent = n;
      if (!rows.length) {
        $('recent').innerHTML =
          '<div class="empty">还没有人提交 —— 你可以是第一个。</div>';
        return;
      }
      const pad = (i) => String(i + 1).padStart(2, '0');
      $('recent').innerHTML = '<div class="feed">' + rows.map((r, i) => {
        const d = new Date(r.created_at);
        const t = isNaN(d.getTime()) ? '' : d.toLocaleString('zh-CN', { hour12: false });
        return '<div class="row">' +
          '<span class="row__idx">' + pad(i) + '</span>' +
          '<span class="row__dist">' + escapeHtml(r.shot_distance_m) + ' m</span>' +
          '<span class="row__main">' + escapeHtml(r.building_note || '（未填写位置说明）') + '</span>' +
          '<span class="row__by">' + escapeHtml(r.contributor_name || '热心网友') + '</span>' +
          '<span class="row__time">' + escapeHtml(t) + '</span>' +
          '</div>';
      }).join('') + '</div>';
    } catch (e) {
      $('recent').innerHTML = '<p class="muted">统计信息暂时无法加载。</p>';
    }
  }

  /* ------------------------------------------------------------------ 启动 */
  async function afterSignIn() {
    const store = window.__CloudStore;
    const { session } = await store.getSession();
    if (!session) {
      showView('auth');
      return;
    }
    renderUser(session);
    showView('main');
    await refreshStats();
  }

  window.addEventListener('DOMContentLoaded', async () => {
    // 图片类型提示
    $('limits-note').textContent =
      `支持 JPG / PNG / WebP，单张不超过 ${fmtSize(L.maxFileBytes)}。`;

    const store = window.__CloudStore;
    if (!store) {
      showView('auth');
      return setMsg($('auth-msg'), '云服务脚本加载失败，请检查网络后刷新页面。', 'err');
    }

    // 公开可读：未登录也能看到已有多少张
    await refreshStats();

    const { session } = await store.getSession();
    if (session) {
      await afterSignIn();
    } else {
      showView('auth');
    }

    window.__CLOUD_READY__.auth.onAuthStateChange(async (event) => {
      if (event === 'SIGNED_OUT') showView('auth');
    });
  });
})();
