document.addEventListener("DOMContentLoaded", () => {
  const statusText = document.getElementById("status-text");
  const headStatusText = document.getElementById("head-status-text");
  const headDirText = document.getElementById("head-direction-text");
  const counterText = document.getElementById("counter-text");
  const mouthCounterText = document.getElementById("mouth-counter-text");
  const totalText = document.getElementById("total-text");
  const alertBox = document.getElementById("alert-box");
  const alertText = document.getElementById("alert-text");
  const beep = document.getElementById("beep-sound");

  // —— CHART SETUP ——
  const ctx = document.getElementById("drowsinessChart").getContext("2d");
  const chartData = {
    labels: [],
    datasets: [
      {
        label: "Total Drowsiness Alerts",
        data: [],
        fill: false,
        borderWidth: 2,
        tension: 0.2,
      },
    ],
  };
  const chartOpts = {
    scales: {
      x: { title: { display: true, text: "Time" } },
      y: { title: { display: true, text: "Alerts Count" }, beginAtZero: true },
    },
  };
  const drowsyChart = new Chart(ctx, {
    type: "line",
    data: chartData,
    options: chartOpts,
  });

  async function fetchStatus() {
    try {
      const res = await fetch("/status");
      const data = await res.json();

      // Overall status
      const status = data.overall ?? data.status ?? "Unknown";
      statusText.textContent = `Status: ${status}`;

      // Head pose
      const down = data.head_down ?? false;
      const left = data.head_left ?? false;
      const right = data.head_right ?? false;
      const oldHead = data.head_direction ?? data.head_status ?? "";
      let headPos = "Normal";
      if (down) headPos = "Down";
      else if (left) headPos = "Turned Left";
      else if (right) headPos = "Turned Right";
      else if (oldHead) headPos = oldHead;

      headStatusText.textContent = `Head Position: ${headPos}`;
      headDirText.textContent = `Head Direction: ${oldHead || headPos}`;

      // Eyes closed & yawning
      const eyesClosed = data.eyes_closed ?? (data.counter ?? 0) > 0;
      const yawning = data.yawning ?? (data.mouth_counter ?? 0) > 0;
      counterText.textContent = `Eyes Closed: ${eyesClosed ? "Yes" : "No"}`;
      mouthCounterText.textContent = `Yawning: ${yawning ? "Yes" : "No"}`;

      // Total alerts
      const totalAlerts = data.total_alerts ?? data.total ?? 0;
      totalText.textContent = `Total Alerts: ${totalAlerts}`;

      // Banner & beep
      if (status !== "Awake") {
        alertBox.classList.add("alert-active");
        alertText.textContent =
          status === "Yawning!"
            ? "😮 Yawning detected!"
            : "😴 Drowsiness detected!";
        if (beep.paused) beep.play();
      } else {
        alertBox.classList.remove("alert-active");
        alertText.textContent = "";
        beep.pause();
        beep.currentTime = 0;
      }

      // Chart update
      const nowCount = totalAlerts;
      const nowLabel = new Date().toLocaleTimeString();
      drowsyChart.data.labels.push(nowLabel);
      drowsyChart.data.datasets[0].data.push(nowCount);
      if (drowsyChart.data.labels.length > 20) {
        drowsyChart.data.labels.shift();
        drowsyChart.data.datasets[0].data.shift();
      }
      drowsyChart.update();
    } catch (err) {
      console.error("Error fetching /status:", err);
    }
  }

  // first fetch & interval
  fetchStatus();
  setInterval(fetchStatus, 500);
});
