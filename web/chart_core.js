(function (global) {
  const CORE_VERSION = "core-20241205-1";
  console.log("[chart_core] load", CORE_VERSION);
  function ensureLib() {
    if (typeof LightweightCharts === "undefined" || typeof LightweightCharts.createChart !== "function") {
      console.error(
        "LightweightCharts not loaded or invalid. 请检查 /vendor/lightweight-charts.standalone.production.js 是否可访问。"
      );
      return false;
    }
    return true;
  }

  function createChart(container, opts) {
    if (!ensureLib()) return null;
    console.log(
      "[chart_core] LightweightCharts.createChart typeof",
      typeof LightweightCharts.createChart
    );
    const chart = LightweightCharts.createChart(container, {
      layout: {
        background: { color: "#0b0f14" },
        textColor: "#e5e7eb",
      },
      grid: {
        vertLines: { color: "#1f2937" },
        horzLines: { color: "#1f2937" },
      },
      crosshair: {
        mode: LightweightCharts.CrosshairMode.Normal,
      },
      rightPriceScale: {
        borderColor: "#374151",
      },
      timeScale: {
        borderColor: "#374151",
      },
      ...opts,
    });

    let candleSeries = null;
    try {
      console.log(
        "[chart_core] chart methods typeof(addSeries):",
        typeof chart.addSeries,
        "CandlestickSeries type:",
        typeof LightweightCharts.CandlestickSeries
      );
      if (typeof chart.addSeries === "function" && LightweightCharts.CandlestickSeries) {
        console.log("[chart_core] use addSeries(CandlestickSeries, options)");
        candleSeries = chart.addSeries(LightweightCharts.CandlestickSeries, {
          upColor: "#10b981",
          downColor: "#ef4444",
          borderUpColor: "#10b981",
          borderDownColor: "#ef4444",
          wickUpColor: "#10b981",
          wickDownColor: "#ef4444",
        });
        console.log(
          "[chart_core] candleSeries created, setData typeof:",
          candleSeries && typeof candleSeries.setData
        );
      }
    } catch (e) {
      console.error("[chart_core] 创建蜡烛图序列失败", e);
      throw e;
    }

    if (!candleSeries) {
      console.error(
        "chart 没有可用的蜡烛图创建方法，请确认 LightweightCharts 版本及用法。chart keys:",
        Object.keys(chart || {})
      );
      return null;
    }

    let volumeSeries = null;
    try {
      console.log(
        "[chart_core] volume addSeries typeof:",
        typeof chart.addSeries,
        "HistogramSeries type:",
        typeof LightweightCharts.HistogramSeries
      );
      if (typeof chart.addSeries === "function" && LightweightCharts.HistogramSeries) {
        console.log("[chart_core] use addSeries(HistogramSeries, options) for volume");
        volumeSeries = chart.addSeries(LightweightCharts.HistogramSeries, {
          priceScaleId: "",
          color: "#60a5fa",
          lineWidth: 2,
          priceFormat: { type: "volume" },
          priceLineVisible: false,
          overlay: true,
        });
        console.log(
          "[chart_core] volumeSeries created, setData typeof:",
          volumeSeries && typeof volumeSeries.setData
        );
      }
    } catch (e) {
      console.error("[chart_core] 创建成交量序列失败", e);
      volumeSeries = null;
    }

    try {
      if (typeof chart.priceScale === "function") {
        console.log("[chart_core] apply main priceScale margins");
        chart.priceScale("").applyOptions({
          scaleMargins: { top: 0.8, bottom: 0 },
        });
      }
    } catch (e) {
      console.warn("[chart_core] 设置 priceScale 失败", e);
    }

    const markerLayers = {};
    let markersPrimitive = null;
    try {
      if (LightweightCharts.createSeriesMarkers && candleSeries) {
        console.log("[chart_core] createSeriesMarkers primitive");
        markersPrimitive = LightweightCharts.createSeriesMarkers(candleSeries, [], {});
      }
    } catch (e) {
      console.warn("[chart_core] 创建 markers primitive 失败，将退回空实现", e);
      markersPrimitive = null;
    }

    return {
      chart,
      candleSeries,
      volumeSeries,
      markerLayers,
      markersPrimitive,
    };
  }

  function setData(handle, bars) {
    if (!handle) {
      console.error("[chart_core] setData 调用时 handle 为空");
      return;
    }
    if (!handle.candleSeries) {
      console.error("[chart_core] setData 时 candleSeries 未初始化", handle);
      return;
    }
    const candleData = bars.map((b) => ({
      time: b.t,
      open: b.o,
      high: b.h,
      low: b.l,
      close: b.c,
    }));
    try {
      console.log("[chart_core] setData 蜡烛数据条数", candleData.length);
      handle.candleSeries.setData(candleData);
    } catch (e) {
      console.error("[chart_core] 设置蜡烛数据失败", e);
      throw e;
    }

    if (handle.volumeSeries) {
      const volumeData = bars.map((b) => ({
        time: b.t,
        value: b.v || 0,
        color: b.c >= b.o ? "rgba(16,185,129,0.5)" : "rgba(239,68,68,0.5)",
      }));
      try {
        console.log("[chart_core] setData 成交量数据条数", volumeData.length);
        handle.volumeSeries.setData(volumeData);
      } catch (e) {
        console.error("[chart_core] 设置成交量数据失败", e);
      }
    }
  }

  function setMarkers(handle, layerId, markers) {
    if (!handle || !handle.candleSeries) {
      console.error("[chart_core] setMarkers 调用时 handle/candleSeries 为空");
      return;
    }
    // markers: [{time, price?, shape?, color?, text?, position?}]
    const formatted = markers.map((m) => ({
      time: m.time,
      position: m.position || (m.shape === "arrowUp" ? "belowBar" : "aboveBar"),
      color: m.color || (m.shape === "arrowUp" ? "#10b981" : "#ef4444"),
      shape: m.shape || "arrowUp",
      text: m.text || "",
      size: m.size || 1,
    }));
    handle.markerLayers[layerId] = formatted;
    // 合并所有 layers
    const allMarkers = Object.values(handle.markerLayers).flat();

    if (handle.markersPrimitive && typeof handle.markersPrimitive.setMarkers === "function") {
      try {
        console.log("[chart_core] 使用 markersPrimitive.setMarkers, 条数:", allMarkers.length);
        handle.markersPrimitive.setMarkers(allMarkers);
      } catch (e) {
        console.error("[chart_core] markersPrimitive.setMarkers 失败", e);
      }
    } else if (typeof handle.candleSeries.setMarkers === "function") {
      try {
        console.log("[chart_core] 回退到 candleSeries.setMarkers, 条数:", allMarkers.length);
        handle.candleSeries.setMarkers(allMarkers);
      } catch (e) {
        console.error("[chart_core] candleSeries.setMarkers 失败", e);
      }
    } else {
      console.warn(
        "[chart_core] 当前系列不支持 setMarkers（既无 primitive 也无法直接 setMarkers），忽略标注。"
      );
    }
  }

  function resetMarkers(handle) {
    if (!handle || !handle.candleSeries) return;
    handle.markerLayers = {};
    if (handle.markersPrimitive && typeof handle.markersPrimitive.clearMarkers === "function") {
      try {
        console.log("[chart_core] resetMarkers 使用 markersPrimitive.clearMarkers");
        handle.markersPrimitive.clearMarkers();
      } catch (e) {
        console.error("[chart_core] markersPrimitive.clearMarkers 失败", e);
      }
    } else if (handle.markersPrimitive && typeof handle.markersPrimitive.setMarkers === "function") {
      try {
        console.log("[chart_core] resetMarkers 使用 markersPrimitive.setMarkers([])");
        handle.markersPrimitive.setMarkers([]);
      } catch (e) {
        console.error("[chart_core] markersPrimitive.setMarkers([]) 失败", e);
      }
    } else if (typeof handle.candleSeries.setMarkers === "function") {
      try {
        console.log("[chart_core] resetMarkers 回退到 candleSeries.setMarkers([])");
        handle.candleSeries.setMarkers([]);
      } catch (e) {
        console.error("[chart_core] candleSeries.setMarkers([]) 失败", e);
      }
    } else {
      console.warn("[chart_core] resetMarkers 时发现无可用的 markers API，直接清空本地状态");
    }
  }

  function autoFit(handle) {
    if (!handle || !handle.chart) return;
    try {
      const ts = handle.chart.timeScale && handle.chart.timeScale();
      if (ts && typeof ts.fitContent === "function") {
        ts.fitContent();
      }
      const ps =
        handle.chart.priceScale &&
        (handle.chart.priceScale("right") || handle.chart.priceScale(""));
      if (ps && typeof ps.applyOptions === "function") {
        // 重新开启自动缩放，让 Y 轴根据当前数据自适应
        ps.applyOptions({ autoScale: true });
      }
    } catch (e) {
      console.error("[chart_core] autoFit 失败", e);
    }
  }

  function setLogScale(handle, enabled) {
    if (!handle || !handle.chart || typeof LightweightCharts === "undefined") return;
    try {
      const scale = handle.chart.priceScale && handle.chart.priceScale("right");
      if (!scale || typeof scale.applyOptions !== "function") return;
      const mode = enabled
        ? LightweightCharts.PriceScaleMode && LightweightCharts.PriceScaleMode.Logarithmic
        : LightweightCharts.PriceScaleMode && LightweightCharts.PriceScaleMode.Normal;
      if (mode === undefined || mode === null) return;
      scale.applyOptions({ mode });
    } catch (e) {
      console.error("[chart_core] setLogScale 失败", e);
    }
  }

  global.ChartCore = {
    createChart,
    setData,
    setMarkers,
    resetMarkers,
    autoFit,
    setLogScale,
    version: CORE_VERSION,
  };
})(window);
