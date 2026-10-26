'use strict';

const $ = (id) => document.getElementById(id);

const state = {
  catalog: null,
  act: false, // standard Robot protocol mode (studio.py --robot-config)
  ws: null,
  ready: false,
  models: [],
  runId: null,
  tiles: new Map(),
  lastJpeg: new Map(),
  history: { p50: [], p99: [] },
  detailId: null,
  detailHasFrame: false,
  lastTelemetry: null,
  awaitingStart: false,
  gatewayTimer: null,
  reconnectTimer: null,
};

const HISTORY = 120;

// -- formatting ------------------------------------------------------------

function fmt(value, digits) {
  return value == null || Number.isNaN(value) ? '–' : Number(value).toFixed(digits);
}

function stateLabel(value) {
  return {
    starting: '启动中',
    connecting: '连接中',
    running: '运行中',
    done: '完成',
    error: '错误',
  }[value] || value || '–';
}

function shortBackend(url) {
  if (!url) return '';
  try {
    return new URL(url).host;
  } catch (error) {
    return String(url);
  }
}

function setText(id, value) {
  $(id).textContent = String(value);
}

function showError(message) {
  const box = $('form-error');
  if (!message) {
    box.hidden = true;
    box.textContent = '';
    return;
  }
  box.hidden = false;
  box.textContent = message;
}

// -- connection ------------------------------------------------------------

function setConn(text, cls) {
  const pill = $('conn');
  pill.textContent = text;
  pill.className = `pill ${cls || ''}`;
}

function connect() {
  const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${scheme}://${location.host}/ws`);
  ws.binaryType = 'arraybuffer';
  state.ws = ws;

  ws.onopen = () => {
    setConn('已连接', 'ok');
    if (state.detailId !== null) send({ type: 'watch', session: state.detailId, on: true });
    if ($('target').value === 'gateway') send({ type: 'models' });
  };
  ws.onclose = () => {
    setConn('已断开，重连中…', 'bad');
    clearTimeout(state.reconnectTimer);
    state.reconnectTimer = setTimeout(connect, 1200);
  };
  ws.onerror = () => setConn('连接错误', 'bad');
  ws.onmessage = onMessage;
}

function send(message) {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify(message));
  }
}

function onMessage(event) {
  if (event.data instanceof ArrayBuffer) {
    onFrame(event.data);
    return;
  }
  let message;
  try {
    message = JSON.parse(event.data);
  } catch (error) {
    return;
  }
  switch (message.type) {
    case 'hello':
      onHello(message);
      break;
    case 'telemetry':
      onTelemetry(message);
      break;
    case 'run_started':
      onRunStarted(message);
      break;
    case 'run_stopped':
      onRunStopped(message);
      break;
    case 'models':
      onModels(message);
      break;
    case 'gateway_metrics':
      onGatewayMetrics(message);
      break;
    case 'error':
      state.awaitingStart = false;
      showError(message.message);
      $('start').disabled = false;
      $('stop').disabled = true;
      $('run-state').textContent = '启动失败';
      break;
    default:
      break;
  }
}

// -- catalog and form ------------------------------------------------------

function onHello(message) {
  state.catalog = message.catalog;
  if (!state.ready) {
    applyCatalog(message.catalog);
    state.ready = true;
  } else {
    $('physics').textContent = message.catalog.physics;
    updateGatewayHint(message.catalog);
  }
}

function applyCatalog(catalog) {
  const defaults = catalog.defaults || {};
  state.act = Boolean(catalog.robot_config);
  document.body.classList.toggle('mode-act', state.act);
  if (state.act) applyRobotCatalog(catalog);
  $('physics').textContent = catalog.physics;
  $('robots').max = String(catalog.limits.max_robots);
  updateGatewayHint(catalog);

  const robotSelect = $('robot-select');
  robotSelect.replaceChildren();
  for (const robot of catalog.robots) {
    if (Array.isArray(robot.physics) && !robot.physics.includes(catalog.physics)) continue;
    const option = document.createElement('option');
    option.value = robot.id;
    option.textContent = robot.label ? `${robot.label} (${robot.id})` : robot.id;
    robotSelect.append(option);
  }

  const suiteSelect = $('suite');
  suiteSelect.replaceChildren();
  for (const name of Object.keys(catalog.suites)) {
    const option = document.createElement('option');
    option.value = name;
    option.textContent = name;
    suiteSelect.append(option);
  }
  if (catalog.suites.libero_spatial) suiteSelect.value = 'libero_spatial';

  if (defaults.robots != null) $('robots').value = defaults.robots;
  if (defaults.duration_s != null) $('duration').value = defaults.duration_s;
  if (defaults.control_hz != null) $('control-hz').value = defaults.control_hz;
  if (defaults.replan_steps != null) $('replan').value = defaults.replan_steps;
  if (defaults.prefetch != null) $('prefetch').value = defaults.prefetch;
  if (defaults.resolution != null) $('resolution').value = String(defaults.resolution);
  if (defaults.frame_fps != null) $('frame-fps').value = defaults.frame_fps;
  if (defaults.seed != null) $('seed').value = defaults.seed;
  if (defaults.base_camera_key) $('base-camera-key').value = defaults.base_camera_key;
  if (defaults.wrist_camera_key) $('wrist-camera-key').value = defaults.wrist_camera_key;
  if (defaults.action_scale != null) $('action-scale').value = defaults.action_scale;
  if (defaults.report_episodes != null) $('report-episodes').checked = Boolean(defaults.report_episodes);

  populateTasks();
}

function applyRobotCatalog(catalog) {
  const robot = catalog.robot_config;
  const defaults = catalog.defaults || {};
  $('target').value = 'gateway';
  $('act-physics').textContent = `${catalog.protocol} · ${catalog.physics}`;
  $('hardware-tag').hidden = !robot.hardware;
  const cameras = Object.entries(robot.cameras).map(([role, kind]) => `${role}=${kind}`).join(', ');
  $('robot-summary').textContent =
    `robot_id ${robot.robot_id} · driver ${robot.driver}${robot.hardware ? '' : '（MuJoCo 孪生）'} · 相机 ${cameras || '无'} · 文件中的模型 ${robot.model}`;
  if (defaults.instruction) $('instruction').value = defaults.instruction;
  if (defaults.mode) $('safety-mode').value = defaults.mode;
  if (defaults.episodes != null) $('episodes').value = defaults.episodes;
  if (defaults.max_steps != null) $('max-steps').value = defaults.max_steps;
  updateConfirmRow();
}

function updateConfirmRow() {
  const hardware = Boolean(state.catalog?.robot_config?.hardware);
  const moves = $('safety-mode').value !== 'dry_run';
  $('confirm-row').hidden = !(hardware && moves);
  if (!moves) $('confirm-motion').checked = false;
}

function updateGatewayHint(catalog) {
  const hint = $('gateway-hint');
  if (!catalog.gateway || !catalog.gateway.url) {
    hint.textContent = '服务端未配置网关，请使用直连模式。';
    return;
  }
  hint.textContent = catalog.gateway.has_key
    ? `服务端：${catalog.gateway.url}`
    : `服务端：${catalog.gateway.url}（未配置 API Key）`;
}

function populateTasks() {
  if (!state.catalog) return;
  const suite = $('suite').value;
  const tasks = state.catalog.suites[suite]?.tasks || [];
  const container = $('tasks');
  container.replaceChildren();
  for (const task of tasks) {
    const label = document.createElement('label');
    const input = document.createElement('input');
    input.type = 'checkbox';
    input.value = String(task.id);
    input.checked = true;
    const text = document.createElement('span');
    text.textContent = `#${task.id} ${task.language}`;
    label.append(input, text);
    container.append(label);
  }
}

function onModels(message) {
  state.models = message.models || [];
  const select = $('model');
  const previous = select.value;
  select.replaceChildren();
  for (const model of state.models) {
    const option = document.createElement('option');
    option.value = model.id;
    option.textContent = model.backends == null ? model.id : `${model.id} (${model.backends} backend)`;
    select.append(option);
  }
  if (previous && state.models.some((model) => model.id === previous)) {
    select.value = previous;
  } else if (state.models.length) {
    select.value = state.models[0].id;
  }
  if (message.error) showError(`无法获取模型列表：${message.error}`);
}

function onGatewayMetrics(message) {
  const model = $('model').value;
  const metrics = message.metrics;
  if (message.error || !metrics || !metrics[model]) {
    setText('m-relay', '–');
    return;
  }
  setText('m-relay', fmt(metrics[model].relay_p50_ms, 0));
}

// -- run control -----------------------------------------------------------

function collectConfig() {
  if (state.act) {
    return {
      model: $('model').value,
      instruction: $('instruction').value.trim(),
      mode: $('safety-mode').value,
      confirm_motion: $('confirm-motion').checked,
      robots: Number($('robots').value),
      episodes: Number($('episodes').value),
      max_steps: Number($('max-steps').value),
      duration_s: Number($('duration').value),
      control_hz: Number($('control-hz').value),
      prefetch: Number($('prefetch').value),
      frame_fps: Number($('frame-fps').value),
    };
  }
  const taskIds = [...document.querySelectorAll('#tasks input:checked')].map((input) => Number(input.value));
  return {
    target: $('target').value,
    model: $('model').value,
    direct_url: $('direct-url').value.trim(),
    robot: $('robot-select').value,
    suite: $('suite').value,
    task_ids: taskIds,
    robots: Number($('robots').value),
    duration_s: Number($('duration').value),
    control_hz: Number($('control-hz').value),
    replan_steps: Number($('replan').value),
    prefetch: Number($('prefetch').value),
    resolution: Number($('resolution').value),
    frame_fps: Number($('frame-fps').value),
    seed: Number($('seed').value),
    base_camera_key: $('base-camera-key').value.trim(),
    wrist_camera_key: $('wrist-camera-key').value.trim(),
    action_scale: Number($('action-scale').value),
    report_episodes: $('report-episodes').checked,
  };
}

function validateConfig(config) {
  if (state.act) {
    if (!config.model) return '请选择或刷新模型（需要网关上配置了 adapter 的模型）。';
    if (!config.instruction) return '请输入指令。';
    if (state.catalog.robot_config.hardware && config.mode !== 'dry_run' && !config.confirm_motion) {
      return `${config.mode} 模式会让真机运动：请先确认工作区已清空、急停在手边。`;
    }
  } else if (!config.task_ids.length) {
    return '至少选择一个任务。';
  }
  if (!Number.isFinite(config.robots) || config.robots < 1) return '机器人数量至少为 1。';
  if (state.catalog && config.robots > state.catalog.limits.max_robots) {
    return `机器人数量最多为 ${state.catalog.limits.max_robots}。`;
  }
  if (!Number.isFinite(config.duration_s) || config.duration_s < 1) return '时长至少为 1 秒。';
  if (config.target === 'gateway' && !config.model) return '请选择或刷新模型。';
  if (config.target === 'direct' && !/^wss?:\/\//.test(config.direct_url)) {
    return '直连地址必须以 ws:// 或 wss:// 开头。';
  }
  return '';
}

function clearTiles() {
  for (const tile of state.tiles.values()) {
    if (tile.img.dataset.blobUrl) URL.revokeObjectURL(tile.img.dataset.blobUrl);
    tile.root.remove();
  }
  state.tiles.clear();
  state.lastJpeg.clear();
}

function startRun() {
  const config = collectConfig();
  const problem = validateConfig(config);
  showError(problem);
  if (problem) return;

  clearTiles();
  closeDetail();
  state.history = { p50: [], p99: [] };
  state.runId = null;
  state.lastTelemetry = null;
  state.awaitingStart = true;
  drawChart();
  $('empty').hidden = true;
  $('start').disabled = true;
  $('stop').disabled = false;
  $('run-state').textContent = '启动中…';
  send({ type: 'start', config });
  startGatewayPolling(state.act || config.target === 'gateway');
}

function stopRun() {
  send({ type: 'stop' });
  $('stop').disabled = true;
  $('run-state').textContent = '停止中…';
  stopGatewayPolling();
}

function onRunStarted(message) {
  state.awaitingStart = false;
  state.runId = message.id;
  $('start').disabled = true;
  $('stop').disabled = false;
  $('run-state').textContent = '运行中…';
}

function onRunStopped() {
  state.awaitingStart = false;
  $('start').disabled = false;
  $('stop').disabled = true;
  $('run-state').textContent = '已停止';
  stopGatewayPolling();
}

// -- telemetry -------------------------------------------------------------

function onTelemetry(message) {
  // A start request is in flight; ignore the previous run's final snapshots
  // until run_started (or an error) tells us which run is current.
  if (state.awaitingStart) return;
  if (state.runId !== message.id) {
    if (state.runId !== null) clearTiles();
    state.runId = message.id;
    state.history = { p50: [], p99: [] };
  }
  state.lastTelemetry = message;
  const global = message.global;
  setText('m-active', `${message.active}/${message.config.robots}`);
  setText('m-rps', fmt(global.throughput_rps, 1));
  setText('m-p50', fmt(global.latency_p50_ms, 0));
  setText('m-p99', fmt(global.latency_p99_ms, 0));
  setText('m-underrun', global.underruns);
  setText('m-late', global.late_tick_pct == null ? '–' : `${global.late_tick_pct.toFixed(1)}%`);
  setText('m-success', global.success_rate == null ? '–' : `${(global.success_rate * 100).toFixed(0)}%`);
  setText('m-episodes', global.episodes);

  pushHistory(global.latency_p50_ms, global.latency_p99_ms);
  for (const session of message.sessions) upsertTile(session);
  if (state.detailId !== null) {
    updateDetail(message.sessions.find((session) => session.id === state.detailId));
  }
  $('empty').hidden = message.sessions.length > 0;
  $('run-state').textContent = message.running
    ? `运行中 · ${message.elapsed_s.toFixed(0)}s`
    : (message.errors ? '已结束（有错误）' : '已结束');
  $('start').disabled = message.running;
  $('stop').disabled = !message.running;
  if (!message.running) stopGatewayPolling();
}

function upsertTile(session) {
  let tile = state.tiles.get(session.id);
  if (!tile) {
    const root = document.createElement('div');
    root.className = 'tile';
    root.dataset.id = String(session.id);

    const thumb = document.createElement('div');
    thumb.className = 'thumb';
    const img = document.createElement('img');
    img.alt = `robot ${session.id}`;
    const badge = document.createElement('span');
    badge.className = 'badge';
    thumb.append(img, badge);

    const body = document.createElement('div');
    body.className = 'tile-body';
    const title = document.createElement('div');
    title.className = 'tile-title';
    const titleText = document.createElement('span');
    const backend = document.createElement('span');
    backend.className = 'backend';
    title.append(titleText, backend);
    const lang = document.createElement('div');
    lang.className = 'tile-lang';
    const stats = document.createElement('div');
    stats.className = 'tile-stats';
    const episodes = document.createElement('span');
    const success = document.createElement('span');
    const latency = document.createElement('span');
    stats.append(episodes, success, latency);
    body.append(title, lang, stats);
    root.append(thumb, body);
    root.addEventListener('click', () => openDetail(session.id));

    tile = { root, img, badge, titleText, backend, lang, episodes, success, latency };
    state.tiles.set(session.id, tile);
    $('grid').append(root);
  }
  tile.titleText.textContent = state.act
    ? `#${session.id} · ${session.robot?.robot_id || '–'} · ${session.robot?.safety?.mode || ''}`
    : `#${session.id} · task ${session.task_id ?? '–'}`;
  tile.backend.textContent = shortBackend(session.backend);
  tile.lang.textContent = session.language || '—';
  tile.lang.title = session.language || '';
  tile.episodes.textContent = `ep ${session.episodes}`;
  tile.success.textContent = state.act
    ? `cmd ${session.robot?.commands_sent ?? 0}`
    : `成功 ${session.successes}`;
  tile.latency.textContent = session.latency_p50_ms == null ? 'p50 –' : `p50 ${session.latency_p50_ms.toFixed(0)} ms`;
  tile.badge.textContent = stateLabel(session.state);
  tile.badge.className = `badge ${session.state}`;
  tile.root.title = session.error || '';
}

function onFrame(buffer) {
  if (buffer.byteLength <= 8) return;
  const view = new DataView(buffer);
  const kind = view.getUint32(0);
  const sessionId = view.getUint32(4);
  if (kind === 2) {
    if (state.detailId === sessionId) {
      state.detailHasFrame = true;
      setImage($('detail-img'), buffer.slice(8));
    }
    return;
  }
  const tile = state.tiles.get(sessionId);
  if (tile) setImage(tile.img, buffer.slice(8));
  state.lastJpeg.set(sessionId, buffer.slice(8));
  if (state.detailId === sessionId && !state.detailHasFrame) {
    setImage($('detail-img'), buffer.slice(8));
  }
}

function setImage(img, arrayBuffer) {
  const url = URL.createObjectURL(new Blob([arrayBuffer], { type: 'image/jpeg' }));
  const previous = img.dataset.blobUrl;
  img.dataset.blobUrl = url;
  img.onload = () => {
    if (previous) URL.revokeObjectURL(previous);
  };
  img.src = url;
}

// -- detail ----------------------------------------------------------------

function watch(sessionId, on) {
  send({ type: 'watch', session: sessionId, on });
}

function openDetail(sessionId) {
  state.detailId = sessionId;
  state.detailHasFrame = false;
  $('detail').hidden = false;
  const img = $('detail-img');
  if (img.dataset.blobUrl) {
    URL.revokeObjectURL(img.dataset.blobUrl);
    delete img.dataset.blobUrl;
  }
  img.removeAttribute('src');
  if (state.lastJpeg.has(sessionId)) setImage(img, state.lastJpeg.get(sessionId));
  watch(sessionId, true);
  updateDetail(state.lastTelemetry?.sessions.find((session) => session.id === sessionId));
}

function closeDetail() {
  if (state.detailId !== null) watch(state.detailId, false);
  state.detailId = null;
  state.detailHasFrame = false;
  const img = $('detail-img');
  if (img.dataset.blobUrl) {
    URL.revokeObjectURL(img.dataset.blobUrl);
    delete img.dataset.blobUrl;
  }
  img.removeAttribute('src');
  $('detail').hidden = true;
}

function updateDetail(session) {
  if (state.detailId === null) return;
  if (!session) {
    $('detail-title').textContent = `#${state.detailId}`;
    return;
  }
  $('detail-title').textContent = state.act
    ? `#${session.id} · ${session.robot?.robot_id || '–'} · ${session.language || '—'}`
    : `#${session.id} · task ${session.task_id ?? '–'} · ${session.language || '—'}`;
  const items = state.act ? robotItems(session) : [
    ['状态', stateLabel(session.state)],
    ['backend', shortBackend(session.backend) || '–'],
    ['模型', session.model || '–'],
    ['episodes', session.episodes],
    ['成功', session.successes],
    ['p50', session.latency_p50_ms == null ? '–' : `${session.latency_p50_ms.toFixed(1)} ms`],
    ['最近延迟', session.last_latency_ms == null ? '–' : `${session.last_latency_ms.toFixed(1)} ms`],
    ['underruns', session.underruns],
    ['ticks', session.ticks],
    ['late ticks', session.late_ticks],
  ];
  const container = $('detail-stats');
  container.replaceChildren();
  for (const [label, value] of items) {
    const item = document.createElement('div');
    item.className = 'item';
    const v = document.createElement('div');
    v.className = 'v';
    v.textContent = String(value);
    const l = document.createElement('div');
    l.className = 'l';
    l.textContent = label;
    item.append(v, l);
    container.append(item);
  }
  if (state.act) $('detail-robot-text').textContent = robotText(session.robot || {});
  const action = session.last_action || [];
  $('detail-action').textContent = action.length ? action.map((value) => value.toFixed(3)).join(', ') : '–';
  const error = $('detail-error');
  if (session.error) {
    error.hidden = false;
    error.textContent = session.error;
  } else {
    error.hidden = true;
    error.textContent = '';
  }
}

function robotItems(session) {
  const robot = session.robot || {};
  const safety = robot.safety || {};
  const ms = (value) => (value == null ? '–' : `${value.toFixed(1)} ms`);
  return [
    ['状态', stateLabel(session.state)],
    ['模型', session.model || '–'],
    ['安全模式', safety.mode || '–'],
    ['episodes', session.episodes],
    ['round trip p50', ms(session.latency_p50_ms)],
    ['server p50', ms(robot.server_p50_ms)],
    ['network p50', ms(robot.network_p50_ms)],
    ['推理次数', robot.inferences ?? 0],
    ['ticks', session.ticks],
    ['已下发命令', robot.commands_sent ?? 0],
    ['clamped', safety.clamped ?? 0],
    ['underruns', session.underruns],
  ];
}

function robotText(robot) {
  const lines = [];
  const add = (label, value) => {
    if (value === undefined) return;
    lines.push(`${label}: ${typeof value === 'object' && value !== null ? JSON.stringify(value) : value}`);
  };
  add('joint_pos (rad)', robot.joint_pos);
  add('gripper (driver units)', robot.gripper_raw);
  add('eef xyz (m)', robot.eef_xyz);
  add('last command', robot.last_command);
  add('safety', robot.safety);
  add('stopped', robot.stopped);
  add('last episode', robot.last_episode);
  add('session', robot.session_id);
  add('control_hz', robot.control_hz);
  add('requires', robot.requires);
  add('provides', robot.provides);
  return lines.join('\n') || '–';
}

// -- chart -----------------------------------------------------------------

function pushHistory(p50, p99) {
  if (p50 != null) state.history.p50.push(p50);
  if (p99 != null) state.history.p99.push(p99);
  while (state.history.p50.length > HISTORY) state.history.p50.shift();
  while (state.history.p99.length > HISTORY) state.history.p99.shift();
  drawChart();
}

function drawChart() {
  const canvas = $('chart');
  const context = canvas.getContext('2d');
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 900;
  const height = 120;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);

  const styles = getComputedStyle(document.documentElement);
  const rule = styles.getPropertyValue('--rule').trim();
  const quiet = styles.getPropertyValue('--quiet').trim();
  const accent = styles.getPropertyValue('--accent').trim();
  const bad = styles.getPropertyValue('--bad').trim();

  const values = [...state.history.p50, ...state.history.p99];
  const controlHz = Number($('control-hz').value) || 0;
  const budget = controlHz > 0 ? 1000 / controlHz : null;
  const maxValue = Math.max(budget || 0, 20, ...values) * 1.15;
  const x = (index, count) => 8 + (index / Math.max(count - 1, 1)) * (width - 16);
  const y = (value) => height - 8 - (value / maxValue) * (height - 16);

  context.strokeStyle = rule;
  context.lineWidth = 1;
  context.beginPath();
  context.moveTo(0, height - 8);
  context.lineTo(width, height - 8);
  context.stroke();

  if (budget) {
    context.setLineDash([4, 4]);
    context.strokeStyle = quiet;
    context.beginPath();
    context.moveTo(0, y(budget));
    context.lineTo(width, y(budget));
    context.stroke();
    context.setLineDash([]);
    context.fillStyle = quiet;
    context.font = '10px system-ui, sans-serif';
    context.fillText(`budget ${budget.toFixed(0)} ms`, Math.max(8, width - 92), y(budget) - 4);
  }

  const line = (points, color) => {
    if (!points.length) return;
    if (points.length === 1) {
      context.fillStyle = color;
      context.beginPath();
      context.arc(x(0, 1), y(points[0]), 2, 0, Math.PI * 2);
      context.fill();
      return;
    }
    context.strokeStyle = color;
    context.lineWidth = 1.6;
    context.beginPath();
    points.forEach((value, index) => {
      const px = x(index, points.length);
      const py = y(value);
      if (index) context.lineTo(px, py);
      else context.moveTo(px, py);
    });
    context.stroke();
  };
  line(state.history.p50, accent);
  line(state.history.p99, bad);

  context.fillStyle = quiet;
  context.font = '10px system-ui, sans-serif';
  context.fillText(`${maxValue.toFixed(0)} ms`, 10, 12);
}

// -- gateway metrics -------------------------------------------------------

function startGatewayPolling(enabled) {
  stopGatewayPolling();
  if (!enabled) {
    setText('m-relay', '–');
    return;
  }
  const poll = () => send({ type: 'gateway_metrics' });
  poll();
  state.gatewayTimer = setInterval(poll, 2000);
}

function stopGatewayPolling() {
  if (state.gatewayTimer) {
    clearInterval(state.gatewayTimer);
    state.gatewayTimer = null;
  }
}

// -- wiring ----------------------------------------------------------------

$('run-form').addEventListener('submit', (event) => {
  event.preventDefault();
  startRun();
});
$('run-form').addEventListener('invalid', (event) => {
  if (event.target && event.target.validationMessage) showError(event.target.validationMessage);
}, true);
$('stop').addEventListener('click', stopRun);
$('refresh-models').addEventListener('click', () => send({ type: 'models' }));
$('suite').addEventListener('change', populateTasks);
$('select-all').addEventListener('click', () => {
  document.querySelectorAll('#tasks input').forEach((input) => { input.checked = true; });
});
$('select-none').addEventListener('click', () => {
  document.querySelectorAll('#tasks input').forEach((input) => { input.checked = false; });
});
$('target').addEventListener('change', () => {
  const direct = $('target').value === 'direct';
  $('gateway-fields').hidden = direct;
  $('direct-fields').hidden = !direct;
});
$('safety-mode').addEventListener('change', updateConfirmRow);
$('detail-close').addEventListener('click', closeDetail);
$('detail').addEventListener('click', (event) => {
  if (event.target === $('detail')) closeDetail();
});
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && state.detailId !== null) closeDetail();
});
window.addEventListener('resize', drawChart);

connect();
drawChart();
