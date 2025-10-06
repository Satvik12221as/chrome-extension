// Example: sending dummy frame data
setInterval(() => {
  const frameData = "test_frame_data"; // Later replace with real base64 from canvas
  chrome.runtime.sendMessage({ type: "FRAME_DATA", data: frameData }, response => {
    console.log("API result:", response);
  });
}, 5000);
