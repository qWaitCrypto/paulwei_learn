(() => {
  const FRONT_VERSION = "main-20241205-1";
  console.log("[main] load", FRONT_VERSION);

  const symbolSelect = document.getElementById("symbol");
  const intervalSelect = document.getElementById("interval");
  const startInput = document.getElementById("start");
  const endInput = document.getElementById("end");
  const loadBtn = document.getElementById("loadBtn");
  const loadTradesBtn = document.getElementById("loadTradesBtn");
  const startPracticeBtn = document.getElementById("startPracticeBtn");
  const revealBtn = document.getElementById("revealBtn");
  const orderBtn = document.getElementById("orderBtn");
  const advanceBtn = document.getElementById("advanceBtn");
  const advance10Btn = document.getElementById("advance10Btn");
  const practiceInfo = document.getElementById("practiceInfo");
  const autoBtn = document.getElementById("autoBtn");
  const logBtn = document.getElementById("logBtn");
  const logBox = document.getElementById("logBox");
  const stateBox = document.getElementById("stateBox");
  const ordersBox = document.getElementById("ordersBox");
  const fillsBox = document.getElementById("fillsBox");

  // 练习配置/下单表单
  const initCashInput = document.getElementById("initCash");
  const feeRateInput = document.getElementById("feeRate");
  const slipPctInput = document.getElementById("slipPct");
  const sideInput = document.getElementById("side");
  const typeInput = document.getElementById("otype");
  const qtyInput = document.getElementById("qty");
  const priceInput = document.getElementById("price");
  const stopPriceInput = document.getElementById("stopPrice");
  const tpInput = document.getElementById("tp");
  const slInput = document.getElementById("sl");

  let chartHandle = null;
  let currentSession = null;
  let isLogScale = false;

  function log(msg) {
    const now = new Date().toISOString();
    logBox.textContent = `[${now}] ${msg}\n` + logBox.textContent;
    console.log("[log]", msg);
  }

  async function fetchJSON(url) {
    const resp = await fetch(url);
    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status}`);
    }
    return resp.json();
  }

  function setDefaultTimeRange() {
    const now = new Date();
    const end = new Date(now.getTime());
    const start = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000); // 近 7 天
    startInput.value = toLocalInput(start);
    endInput.value = toLocalInput(end);
  }

  function toLocalInput(d) {
    // YYYY-MM-DDTHH:mm for datetime-local
    const pad = (n) => String(n).padStart(2, "0");
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(
      d.getHours()
    )}:${pad(d.getMinutes())}`;
  }

  function toISOStringFromInput(val) {
    if (!val) return null;
    const d = new Date(val);
    return d.toISOString();
  }

  async function loadSymbols() {
    const data = await fetchJSON("/api/symbols");
    (data.symbols || []).forEach((s) => {
      const opt = document.createElement("option");
      opt.value = s;
      opt.textContent = s;
      symbolSelect.appendChild(opt);
    });
    if (data.symbols && data.symbols.length > 0) {
      symbolSelect.value = "XBTUSD";
    }
    log(`Loaded ${data.symbols.length} symbols.`);
  }

  function ensureChart() {
    if (!chartHandle) {
      const container = document.getElementById("chart");
      chartHandle = ChartCore.createChart(container, {});
      if (!chartHandle) {
        throw new Error("图表初始化失败，请检查 Lightweight Charts 脚本是否正确加载");
      }
    }
    return chartHandle;
  }

  async function loadKlines() {
    const symbol = symbolSelect.value;
    const interval = intervalSelect.value;
    const startIso = toISOStringFromInput(startInput.value);
    const endIso = toISOStringFromInput(endInput.value);
    if (!startIso || !endIso) {
      log("请先填写开始/结束时间");
      return;
    }
    const params = new URLSearchParams({
      symbol,
      interval,
      start: startIso,
      end: endIso,
      columns: "ohlcv",
    });
    const qs = params.toString();
    const url = `/api/kline?${qs}`;
    // 额外输出 URL 的字符编码，排查是否有奇怪字符
    const urlCharCodes = Array.from(url).map((ch) => ch.charCodeAt(0));
    console.log("[debug] params", qs, "url", url, "charCodes", urlCharCodes.join(","));
    log(`请求 K 线: ${url}`);
    const data = await fetchJSON(url);
    if (data.s !== "ok" || !data.bars) {
      log(`未获取到数据 s=${data.s}`);
      return;
    }
    const bars = data.bars.map((b) => ({
      t: b.t,
      o: b.o,
      h: b.h,
      l: b.l,
      c: b.c,
      v: b.v,
    }));
    ensureChart();
    ChartCore.setData(chartHandle, bars);
    ChartCore.autoFit(chartHandle);
    ChartCore.resetMarkers(chartHandle);
    log(`已加载 ${bars.length} 根 K 线`);
  }

  async function startPractice() {
    log("创建练习 session（随机品种/时间段）...");
    const body = {
      symbol: symbolSelect.value || null,
      interval: intervalSelect.value || "15m",
      duration_minutes: 24 * 60,
      initial_cash: parseFloat(initCashInput.value) || 100000,
      fee_rate: parseFloat(feeRateInput.value) || 0.0006,
      slippage_pct: parseFloat(slipPctInput.value) || 0.0005,
    };
    const resp = await fetch("/api/practice/session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status}`);
    }
    const session = await resp.json();
    currentSession = session;
    practiceInfo.textContent = `Session: ${session.session_id} | ${session.symbol} ${session.interval} | ${session.start} ~ ${session.end}`;
    // 更新输入框以便用户看到范围
    startInput.value = toLocalInput(new Date(session.start));
    endInput.value = toLocalInput(new Date(session.end));
    log(`练习 session 创建成功: ${session.session_id}, ${session.symbol} ${session.interval}`);
    await refreshPracticeState();
  }

  async function revealPaulTrades() {
    if (!currentSession) {
      log("请先开始练习，再揭晓 Paul 成交");
      return;
    }
    const url = `/api/practice/paul_trades?session_id=${encodeURIComponent(currentSession.session_id)}`;
    log(`请求揭晓 Paul 成交: ${url}`);
    const data = await fetchJSON(url);
    if (data.s !== "ok" || !data.trades) {
      log(`未获取到 Paul 成交 s=${data.s}`);
      return;
    }
    const markers = data.trades.map((t) => {
      const timeSec = Math.floor(Date.parse(t.timestamp) / 1000);
      const isBuy = (t.side || "").toLowerCase() === "buy";
      return {
        time: timeSec,
        position: isBuy ? "belowBar" : "aboveBar",
        color: isBuy ? "#10b981" : "#ef4444",
        shape: isBuy ? "arrowUp" : "arrowDown",
        text: `${t.side || ""} ${t.qty || ""}@${t.price}`,
      };
    });
    ensureChart();
    ChartCore.setMarkers(chartHandle, "paul_trades", markers);
    log(`揭晓 Paul 成交标注 ${markers.length} 条`);
  }

  async function loadTrades() {
    const symbol = symbolSelect.value;
    const startIso = toISOStringFromInput(startInput.value);
    const endIso = toISOStringFromInput(endInput.value);
    if (!startIso || !endIso) {
      log("请先填写开始/结束时间");
      return;
    }
    const params = new URLSearchParams({
      symbol,
      start: startIso,
      end: endIso,
    });
    const url = `/api/trades?${params.toString()}`;
    log(`请求 Paul 成交: ${url}`);
    const data = await fetchJSON(url);
    if (data.s !== "ok" || !data.trades) {
      log(`未获取到成交 s=${data.s}`);
      return;
    }
    const markers = data.trades.map((t) => {
      const timeSec = Math.floor(Date.parse(t.timestamp) / 1000);
      const isBuy = (t.side || "").toLowerCase() === "buy";
      return {
        time: timeSec,
        position: isBuy ? "belowBar" : "aboveBar",
        color: isBuy ? "#10b981" : "#ef4444",
        shape: isBuy ? "arrowUp" : "arrowDown",
        text: `${t.side || ""} ${t.qty || ""}@${t.price}`,
      };
    });
    ensureChart();
    ChartCore.setMarkers(chartHandle, "paul_trades", markers);
    log(`已加载 Paul 成交标注 ${markers.length} 条`);
  }

  async function refreshPracticeState() {
    if (!currentSession) return;
    const url = `/api/practice/state?session_id=${encodeURIComponent(currentSession.session_id)}`;
    const state = await fetchJSON(url);
    ensureChart();
    const bars = (state.bars || []).map((b) => ({
      t: b.t,
      o: b.o,
      h: b.h,
      l: b.l,
      c: b.c,
      v: b.v,
    }));
    ChartCore.setData(chartHandle, bars);
    ChartCore.autoFit(chartHandle);
    // 更新状态显示
    stateBox.textContent = JSON.stringify(
      {
        equity: state.equity,
        cash: state.cash,
        realized_pnl: state.realized_pnl,
        position: state.position,
        current_index: state.current_index,
        total_bars: state.total_bars,
      },
      null,
      2
    );
    ordersBox.textContent = JSON.stringify(state.open_orders || [], null, 2);
    fillsBox.textContent = JSON.stringify(state.fills || [], null, 2);
  }

  async function advancePractice(steps) {
    if (!currentSession) {
      log("请先开始练习");
      return;
    }
    const url = `/api/practice/advance?session_id=${encodeURIComponent(
      currentSession.session_id
    )}&steps=${steps}`;
    const resp = await fetch(url, { method: "POST" });
    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status}`);
    }
    const state = await resp.json();
    // 更新图表和状态
    ensureChart();
    const bars = (state.bars || []).map((b) => ({
      t: b.t,
      o: b.o,
      h: b.h,
      l: b.l,
      c: b.c,
      v: b.v,
    }));
    ChartCore.setData(chartHandle, bars);
    ChartCore.autoFit(chartHandle);
    stateBox.textContent = JSON.stringify(
      {
        equity: state.equity,
        cash: state.cash,
        realized_pnl: state.realized_pnl,
        position: state.position,
        current_index: state.current_index,
        total_bars: state.total_bars,
      },
      null,
      2
    );
    ordersBox.textContent = JSON.stringify(state.open_orders || [], null, 2);
    fillsBox.textContent = JSON.stringify(state.fills || [], null, 2);
  }

  async function submitOrder() {
    if (!currentSession) {
      log("请先开始练习");
      return;
    }
    const payload = {
      side: sideInput.value,
      type: typeInput.value,
      qty: parseFloat(qtyInput.value),
      price: priceInput.value ? parseFloat(priceInput.value) : null,
      stop_price: stopPriceInput.value ? parseFloat(stopPriceInput.value) : null,
      take_profit: tpInput.value ? parseFloat(tpInput.value) : null,
      stop_loss: slInput.value ? parseFloat(slInput.value) : null,
    };
    // 修正 stop_loss 输入 ID
    payload.stop_loss = slInput.value ? parseFloat(slInput.value) : null;
    const resp = await fetch(`/api/practice/order?session_id=${encodeURIComponent(currentSession.session_id)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!resp.ok) {
      const text = await resp.text();
      log(`下单失败: ${text}`);
      return;
    }
    const state = await resp.json();
    ensureChart();
    const bars = (state.bars || []).map((b) => ({
      t: b.t,
      o: b.o,
      h: b.h,
      l: b.l,
      c: b.c,
      v: b.v,
    }));
    ChartCore.setData(chartHandle, bars);
    ChartCore.autoFit(chartHandle);
    stateBox.textContent = JSON.stringify(
      {
        equity: state.equity,
        cash: state.cash,
        realized_pnl: state.realized_pnl,
        position: state.position,
        current_index: state.current_index,
        total_bars: state.total_bars,
      },
      null,
      2
    );
    ordersBox.textContent = JSON.stringify(state.open_orders || [], null, 2);
    fillsBox.textContent = JSON.stringify(state.fills || [], null, 2);
    log("下单成功");
  }

  async function init() {
    log(`前端版本 ${FRONT_VERSION}, ChartCore ${ChartCore ? ChartCore.version : "n/a"}`);
    if (typeof LightweightCharts === "undefined") {
      log("LightweightCharts 未加载，请检查 vendor 脚本路径。");
    } else {
      log("LightweightCharts 已加载，API keys: " + Object.keys(LightweightCharts).join(","));
    }
    setDefaultTimeRange();
    await loadSymbols();
    await loadKlines();
  }

  loadBtn.addEventListener("click", () => {
    loadKlines().catch((err) => log(`加载 K 线失败: ${err}`));
  });
  loadTradesBtn.addEventListener("click", () => {
    loadTrades().catch((err) => log(`加载成交失败: ${err}`));
  });
  startPracticeBtn.addEventListener("click", () => {
    startPractice().catch((err) => log(`练习模式失败: ${err}`));
  });
  revealBtn.addEventListener("click", () => {
    revealPaulTrades().catch((err) => log(`揭晓失败: ${err}`));
  });
  advanceBtn.addEventListener("click", () => {
    advancePractice(1).catch((err) => log(`推进失败: ${err}`));
  });
  advance10Btn.addEventListener("click", () => {
    advancePractice(10).catch((err) => log(`推进失败: ${err}`));
  });
  orderBtn.addEventListener("click", () => {
    submitOrder().catch((err) => log(`下单失败: ${err}`));
  });

  if (autoBtn) {
    autoBtn.addEventListener("click", () => {
      try {
        ensureChart();
        ChartCore.autoFit(chartHandle);
      } catch (err) {
        log(`Auto 适配失败: ${err}`);
      }
    });
  }

  if (logBtn) {
    logBtn.addEventListener("click", () => {
      try {
        ensureChart();
        isLogScale = !isLogScale;
        ChartCore.setLogScale(chartHandle, isLogScale);
        logBtn.classList.toggle("active", isLogScale);
        logBtn.title = isLogScale ? "Log: 对数坐标已开启" : "Log: 对数坐标已关闭";
        log(isLogScale ? "Log 对数坐标: 开" : "Log 对数坐标: 关");
      } catch (err) {
        log(`切换 Log 失败: ${err}`);
      }
    });
  }

  init().catch((err) => log(`初始化失败: ${err}`));
})();
