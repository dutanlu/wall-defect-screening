/* ==========================================================================
 * 云服务接入层
 * SDK 形态：CDN <script>（本项目是纯静态站点，无构建步骤）
 * 全局对象：WorkBuddyCloud
 *
 * 关键点：
 *  1. createWorkBuddyCloud 必须同时传 endpoint 与 publishableKey，二者只能来自
 *     publicConfig（config.js），不能硬编码、不能从 location / env 推断。
 *  2. 客户端只初始化一次，Auth / Database / Storage 共用同一个实例。
 *  3. 登录后，共享请求层会自动把当前会话附带到 Database / Storage 请求上，
 *     业务代码不需要手动传 token，也绝不要手动传 owner_id。
 * ========================================================================== */
(function () {
  'use strict';

  const cfg = window.APP_PUBLIC_CONFIG;

  if (!window.WorkBuddyCloud) {
    console.error('[cloud] WorkBuddyCloud 全局对象未就绪，检查 CDN 脚本是否加载成功');
    window.__CLOUD_READY__ = null;
    return;
  }

  const cloud = window.WorkBuddyCloud.createWorkBuddyCloud({
    endpoint: cfg.endpoint,
    publishableKey: cfg.publishableKey,
  });

  window.__CLOUD_READY__ = cloud;
  window.__CloudStore = {
    cloud,
    CloudStoragePathError: window.WorkBuddyCloud.CloudStoragePathError,
    CloudStorageError: window.WorkBuddyCloud.CloudStorageError,

    /* ---- 会话 ---- */
    async getSession() {
      const res = await cloud.auth.getSession();
      if (res && res.error) return { session: null, error: res.error };
      return { session: (res && res.data) || null, error: null };
    },

    /* ---- 数据库：新增一条投递记录，返回插入后的行 ---- */
    async insertSubmission(row) {
      const { data, error } = await cloud.database
        .from('photo_submissions')
        .insert(row)
        .select();
      if (error) throw error;
      return (data && data[0]) || null;
    },

    /* ---- 数据库：读最近 N 条（公开可读，用于展示"已有 N 张"） ---- */
    async listSubmissions(limit) {
      const { data, error } = await cloud.database
        .from('photo_submissions')
        .select('id, contributor_name, building_note, shot_distance_m, created_at')
        .order('created_at', { ascending: false })
        .limit(limit || 12);
      if (error) throw error;
      return data || [];
    },

    /* ---- 数据库：只取总数 ---- */
    async countSubmissions() {
      const { count, error } = await cloud.database
        .from('photo_submissions')
        .select('*', { count: 'exact', head: true });
      if (error) throw error;
      return count || 0;
    },

    /* ---- 存储：上传到 users/<uid>/（user_private）
     * 为什么用 userPath 而不是 sharedPath：
     *   本项目的照片是网友热心投稿的**私人素材**，不应让其他登录网友互相看到。
     *   两种 scope 的可见性（官方 SDK）：
     *     users/<uid>/...   → 仅上传者本人 + 应用管理员可读
     *     shared/<uid>/...  → **所有已登录用户** + 管理员可读
     *   注意：SDK 只有唯一 logical bucket `runtime`，**没有「切换桶可见性」这个能力**
     *   （也不支持 public bucket / public URL），所以「改成私有桶」的说法不准确，
     *   真正的动作是「换 scope」。
     *   DB 里的 photo_key 存的是完整逻辑路径，读取端按该字段原样使用，
     *   不要在前端重新拼前缀。
     * ------------------------------------------------------------------------ */
    async uploadPhoto(session, file) {
      const ext = (file.name.split('.').pop() || 'jpg').toLowerCase().replace(/[^a-z0-9]/g, '');
      const uuid = (crypto.randomUUID ? crypto.randomUUID() : String(Date.now() + Math.random()))
        .replace(/-/g, '');
      const path = cloud.storage.userPath(session.user.id, `facade/${uuid}.${ext}`);
      const res = await cloud.storage.upload(path, file, {
        contentType: file.type,
        cacheControl: '3600',
        upsert: false,
        metadata: { purpose: 'facade-photo' },
      });
      if (res && res.error) throw res.error;
      return path;
    },

    /* ---- 存储：换取临时可访问链接（默认 600 秒） ---- */
    async signedUrl(path, ttlSeconds) {
      const res = await cloud.storage.createSignedUrl(path, ttlSeconds || 600);
      if (res && res.error) throw res.error;
      return (res && res.data && (res.data.signedUrl || res.data.url)) || null;
    },
  };
})();
