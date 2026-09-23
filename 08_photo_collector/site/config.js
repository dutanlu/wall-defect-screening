/* ==========================================================================
 * 站点公开配置（本文件随前端一起发布，可被浏览器看到，属于设计内行为）
 * 来源：workbuddy_cloud_service 返回的 publicConfig
 * 注意：publishableKey 只用于标识"是哪个应用"，本身不带任何权限；
 *       服务端会强制校验访问来源域名（Origin）。密钥类信息不在前端。
 * ========================================================================== */
window.APP_PUBLIC_CONFIG = {
  resourceId: 'wbcs_r6al4Lj3HLytK9itlTs7zA',
  endpoint: 'https://facade-photo-collect.app.workbuddy.host',
  publishableKey: 'wbpk_W69yHHyVCcBl0Sqk0o9m0l_W0URnWXRvvxERbnWtQpzzV3b9780AZec',
  applicationId: 'wbapp_W69yHHyVCcBl0Sqk0o9m0l',
};

/* 上传约束 */
window.APP_LIMITS = {
  maxFileBytes: 15 * 1024 * 1024, // 单张 15 MB
  allowedTypes: ['image/jpeg', 'image/png', 'image/webp'],
  minDistance: 0.1, // 米
  maxDistance: 500, // 米
};
