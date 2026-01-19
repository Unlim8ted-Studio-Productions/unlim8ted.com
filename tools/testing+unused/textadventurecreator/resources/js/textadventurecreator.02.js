// Load different node types and attach event listeners for them
        window.addEventListener('DOMContentLoaded', function () {
        
          // Choice Node with dynamic output
          function createChoiceNode() {
              var node = Ned.createNode("Choice Node", [], ["Choice 1"], 300, 200);
              
              // Button to add more choices dynamically
              var addOutputButton = document.createElementNS(editor.svg.ns, "foreignObject");
              addOutputButton.setAttribute("width", 120);
              addOutputButton.setAttribute("height", 30);
              addOutputButton.setAttribute("x", "30");
              addOutputButton.setAttribute("y", `${node.size.height - 30}`); // Positioned at the bottom
              addOutputButton.innerHTML = '<button>Add Choice</button>';
              node.eRoot.appendChild(addOutputButton);
        
              // Add more outputs when button is clicked
              addOutputButton.querySelector('button').addEventListener('click', () => {
                  var outputName = "Choice " + (node.outputs.length + 1);
                  node.addOutput(outputName);
        
                  // Resize node to fit the new choices (expanding by a controlled amount)
                  node.size = { width: node.size.width, height: node.size.height + 20 }; // Controlled expansion
                  addOutputButton.setAttribute("y", `${node.size.height - 30}`); // Move button to the new bottom
                  node.updateVisuals();
              });
          }
        
          // Logic Node (if-else)
          function createLogicNode() {
              createNode("Logic Node", ["Condition"], ["True", "False"], 400, 100);
          }
        
          // Reward Node
          function createRewardNode() {
              createNode("Reward Node", [], ["Reward"], 200, 100);
          }
        
          // Increase/Decrease Node
          function createIncreaseNode() {
              createNode("Increase/Decrease", ["Value"], ["Modified"], 150, 150);
          }
        
          // Location Node
          function createLocationNode() {
              createNode("Location Node", [], ["Location Output"], 350, 250);
          }
        
          // Character Node
          function createCharacterNode() {
              createNode("Character Node", [], ["Character Info"], 450, 250);
          }
        
          // Item Node
          function createItemNode() {
              createNode("Item Node", [], ["Item Info"], 500, 150);
          }
        
          // Text Node
          function createTextNode() {
              createNode("Text Node", [], ["Next"], 400, 200);
          }
        
          // Attach event listeners to buttons in HTML
          document.querySelector("#addChoiceNode").addEventListener("click", createChoiceNode);
          document.querySelector("#addLogicNode").addEventListener("click", createLogicNode);
          document.querySelector("#addRewardNode").addEventListener("click", createRewardNode);
          document.querySelector("#addIncreaseNode").addEventListener("click", createIncreaseNode);
          document.querySelector("#addLocationNode").addEventListener("click", createLocationNode);
          document.querySelector("#addCharacterNode").addEventListener("click", createCharacterNode);
          document.querySelector("#addItemNode").addEventListener("click", createItemNode);
          document.querySelector("#addTextNode").addEventListener("click", createTextNode);
        
          // Delete functionality
          document.querySelector("#deleteNode").addEventListener("click", () => {
              if (editor.selectedNodes.length > 0) {
                  for (let node of editor.selectedNodes) {
                      node.destroy();
                  }
                  editor.selectedNodes = [];
              }
          });
        
          // Delete with "Delete" key
          document.addEventListener("keydown", function(event) {
              if (event.key === "Delete" || event.key === "Backspace") {
                  if (editor.selectedNodes.length > 0) {
                      for (let node of editor.selectedNodes) {
                          node.destroy();
                      }
                      editor.selectedNodes = [];
                  }
              }
          });
        
          // Export to XML
          function exportToXML() {
              let xml = '<nodes>\n';
              editor.nodegroup.childNodes.forEach((nodeElement) => {
                if (nodeElement.tagName === 'svg') {
                    let title = nodeElement.querySelector('text').textContent;
                    let x = nodeElement.getAttribute('x');
                    let y = nodeElement.getAttribute('y');
                    xml += `  <node title="${title}" x="${x}" y="${y}">\n`;
        
                    // Export inputs and outputs
                    let inputs = nodeElement.querySelectorAll('.Inputs text');
                    let outputs = nodeElement.querySelectorAll('.Outputs text');
                    inputs.forEach((input) => {
                        xml += `    <input name="${input.textContent}" />\n`;
                    });
                    outputs.forEach((output) => {
                        xml += `    <output name="${output.textContent}" />\n`;
                    });
        
                    xml += '  </node>\n';
                }
            });
            xml += '</nodes>';
            console.log(xml);
            // Optionally, download the XML as a file
            downloadXML(xml);
        }
        
        function downloadXML(xml) {
            const blob = new Blob([xml], { type: 'text/xml' });
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = 'nodes.xml';
            link.click();
        }
        
        document.querySelector("#exportXML").addEventListener("click", exportToXML);
        });
