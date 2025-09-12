'use strict';

/// script-runner.js
/// alias sr.js
/// world PAGE
/// dependency run-at.fn
// *##+js(sr)
function containerpopup()
{
let container, minimized = false;
  let scripts = JSON.parse(localStorage.getItem("unlim8tedScripts") || "[]");

  // --- Helper: Save scripts ---
  function saveScripts() {
    localStorage.setItem("unlim8tedScripts", JSON.stringify(scripts));
  }

  // --- Custom modal (replaces alert/prompt) ---
  function customModal(message, withInput = false, callback = null, defaultValue = "") {
    const overlay = document.createElement("div");
    overlay.style.cssText = `
      position: fixed; top: 0; left: 0; width: 100%; height: 100%;
      background: rgba(0,0,0,0.6); display: flex; align-items: center; justify-content: center;
      z-index: 1000000; font-family: sans-serif;
    `;
    const modal = document.createElement("div");
    modal.style.cssText = `
      background: #1e1e1e; padding: 20px; border-radius: 10px;
      color: #fff; width: 300px; box-shadow: 0 6px 20px rgba(0,0,0,0.4);
    `;
    const msg = document.createElement("div");
    msg.style.marginBottom = "12px";
    msg.textContent = message;
    modal.appendChild(msg);

    let input;
    if (withInput) {
      input = document.createElement("input");
      input.type = "text";
      input.value = defaultValue;
      input.style.cssText = `
        width: 100%; padding: 6px; margin-bottom: 12px; border-radius: 6px;
        border: none; background: #2a2a2a; color: #fff;
      `;
      modal.appendChild(input);
    }

    const btnRow = document.createElement("div");
    btnRow.style.cssText = "display:flex; justify-content:flex-end; gap:8px;";
    const okBtn = document.createElement("button");
    okBtn.textContent = "OK";
    okBtn.style.cssText = `
      padding: 4px 10px; border-radius: 6px; border: none;
      background: #4caf50; color: white; cursor: pointer;
    `;
    okBtn.onclick = () => {
      document.body.removeChild(overlay);
      if (callback) callback(withInput ? input.value : true);
    };
    const cancelBtn = document.createElement("button");
    cancelBtn.textContent = "Cancel";
    cancelBtn.style.cssText = `
      padding: 4px 10px; border-radius: 6px; border: none;
      background: #f44336; color: white; cursor: pointer;
    `;
    cancelBtn.onclick = () => document.body.removeChild(overlay);

    btnRow.appendChild(cancelBtn);
    btnRow.appendChild(okBtn);
    modal.appendChild(btnRow);

    overlay.appendChild(modal);
    document.body.appendChild(overlay);
    if (input) input.focus();
  }

  function createContainer() {
    if (container) return;

    container = document.createElement("div");
    container.id = "unlim8ted-js-console";
    container.style.cssText = `
      position: fixed; bottom: 20px; right: 20px;
      width: 460px; height: 380px;
      background: #1e1e1e; color: #fff; font-family: sans-serif;
      border-radius: 12px; box-shadow: 0 8px 20px rgba(0,0,0,0.35);
      display: flex; flex-direction: column; z-index: 999999;
    `;

    // --- Title bar ---
    const titleBar = document.createElement("div");
    titleBar.style.cssText = `
      background: #333; padding: 6px 10px; display: flex;
      justify-content: space-between; align-items: center;
      font-weight: bold; border-top-left-radius: 12px; border-top-right-radius: 12px;
      cursor: move; user-select: none;
    `;
    const titleLeft = document.createElement("span");
    titleLeft.textContent = "JavaScript Runner";
    const btnGroupTop = document.createElement("div");
    btnGroupTop.style.cssText = "display:flex; gap:6px;";
    const minimizeBtn = document.createElement("button");
    minimizeBtn.textContent = "–";
    minimizeBtn.style.cssText = `
      background: none; border: none; color: #fff; font-size: 18px; cursor: pointer;
    `;
    minimizeBtn.onclick = () => {
      minimized = !minimized;
      editor.style.display = minimized ? "none" : "flex";
      buttons.style.display = minimized ? "none" : "flex";
      historySection.style.display = minimized ? "none" : "block";
      container.style.height = minimized ? "40px" : "380px";
    };
    const closeBtn = document.createElement("button");
    closeBtn.textContent = "×";
    closeBtn.style.cssText = `
      background: none; border: none; color: #fff; font-size: 18px; cursor: pointer;
    `;
    closeBtn.onclick = () => {
      document.body.removeChild(container);
      container = null;
    };
    btnGroupTop.appendChild(minimizeBtn);
    btnGroupTop.appendChild(closeBtn);

    titleBar.appendChild(titleLeft);
    titleBar.appendChild(btnGroupTop);

    // --- Editor ---
    const editor = document.createElement("textarea");
    editor.style.cssText = `
      flex: 1; resize: none; border: none; outline: none;
      background: #1e1e1e; color: #dcdcdc; font-family: monospace;
      font-size: 14px; padding: 10px; border-bottom: 1px solid #333;
    `;

    // --- Buttons row with branding ---
    const buttons = document.createElement("div");
    buttons.style.cssText = `
      display: flex; justify-content: space-between; align-items: center;
      padding: 6px 10px; gap: 6px; background: #2c2c2c;
    `;
    const btnRow = document.createElement("div");
    btnRow.style.cssText = "display:flex; gap:6px;";
    const runBtn = document.createElement("button");
    runBtn.textContent = "Run";
    runBtn.style.cssText = `
      padding: 4px 10px; border-radius: 6px; border: none;
      background: #4caf50; color: white; cursor: pointer;
    `;
    runBtn.onclick = () => {
      try {
        const script = document.createElement("script");
        script.textContent = editor.value;
        document.body.appendChild(script);
        document.body.removeChild(script);
      } catch (e) {
        customModal("Error: " + e.message);
      }
    };
    const saveBtn = document.createElement("button");
    saveBtn.textContent = "Save";
    saveBtn.style.cssText = `
      padding: 4px 10px; border-radius: 6px; border: none;
      background: #2196f3; color: white; cursor: pointer;
    `;
    saveBtn.onclick = () => {
      customModal("Enter script name:", true, (name) => {
        if (name) {
          scripts.push({ name, code: editor.value });
          saveScripts();
          renderHistory();
        }
      }, "Script " + (scripts.length + 1));
    };
    btnRow.appendChild(runBtn);
    btnRow.appendChild(saveBtn);

    const branding = document.createElement("a");
    branding.href = "https://unlim8ted.com";
    branding.target = "_blank";
    branding.textContent = "Made by Unlim8ted Studios";
    branding.style.cssText = "font-size:12px; opacity:0.8; color:#fff; text-decoration:none;";

    buttons.appendChild(btnRow);
    buttons.appendChild(branding);

    // --- History section ---
    const historySection = document.createElement("div");
    historySection.style.cssText = `
      background: #1c1c1c; padding: 5px 10px; max-height: 120px;
      overflow-y: auto; border-top: 1px solid #333;
    `;
    const historyTitle = document.createElement("div");
    historyTitle.textContent = "Saved Scripts ▼";
    historyTitle.style.cssText = "cursor:pointer; font-weight:bold; margin-bottom:4px;";
    let collapsed = true;
    const historyList = document.createElement("div");
    historyList.style.display = "none";
    historyTitle.onclick = () => {
      collapsed = !collapsed;
      historyList.style.display = collapsed ? "none" : "block";
      historyTitle.textContent = collapsed ? "Saved Scripts ▼" : "Saved Scripts ▲";
    };
    historySection.appendChild(historyTitle);
    historySection.appendChild(historyList);

    function renderHistory() {
      historyList.innerHTML = "";
      scripts.forEach((item, i) => {
        const row = document.createElement("div");
        row.style.cssText = "display:flex; gap:4px; align-items:center; margin-bottom:2px;";

        const input = document.createElement("input");
        input.type = "text";
        input.value = item.name;
        input.style.cssText = `
          flex:1; background:#2a2a2a; color:#fff; border:none;
          padding:2px 4px; border-radius:4px; font-size:13px;
        `;
        input.onchange = () => {
          scripts[i].name = input.value;
          saveScripts();
        };

        const loadBtn = document.createElement("button");
        loadBtn.textContent = "Load";
        loadBtn.style.cssText = `
          background:#444; border:none; color:#fff;
          padding:2px 6px; border-radius:4px; cursor:pointer; font-size:12px;
        `;
        loadBtn.onclick = () => editor.value = item.code;

        const delBtn = document.createElement("button");
        delBtn.textContent = "×";
        delBtn.style.cssText = `
          background:#b71c1c; border:none; color:#fff;
          padding:2px 6px; border-radius:4px; cursor:pointer; font-size:12px;
        `;
        delBtn.onclick = () => {
          customModal("Delete this script?", false, (ok) => {
            if (ok) {
              scripts.splice(i, 1);
              saveScripts();
              renderHistory();
            }
          });
        };

        row.appendChild(input);
        row.appendChild(loadBtn);
        row.appendChild(delBtn);
        historyList.appendChild(row);
      });
    }
    renderHistory();

    // --- Assemble ---
    container.appendChild(titleBar);
    container.appendChild(editor);
    container.appendChild(buttons);
    container.appendChild(historySection);
    document.body.appendChild(container);

    // --- Make draggable ---
    let offsetX, offsetY, dragging = false;
    titleBar.addEventListener("mousedown", (e) => {
      dragging = true;
      offsetX = e.clientX - container.getBoundingClientRect().left;
      offsetY = e.clientY - container.getBoundingClientRect().top;
      document.addEventListener("mousemove", onDrag);
      document.addEventListener("mouseup", onStopDrag);
    });
    function onDrag(e) {
      if (!dragging) return;
      container.style.left = e.clientX - offsetX + "px";
      container.style.top = e.clientY - offsetY + "px";
      container.style.right = "auto";
      container.style.bottom = "auto";
    }
    function onStopDrag() {
      dragging = false;
      document.removeEventListener("mousemove", onDrag);
      document.removeEventListener("mouseup", onStopDrag);
    }
  }

    document.addEventListener("keydown", (e) => {
    if (e.ctrlKey && e.shiftKey && (e.key === "`" || e.keyCode === 192)) {
      e.preventDefault();
      createContainer();
    }
  });
}
