var grid = document.querySelector("#grid");

		var editor = Ned.create("#svg");
		editor.snapping = 0;

		editor.panZoom = svgPanZoom(editor.svg, {
			viewportSelector: ".svg-pan-zoom_viewport", 
			panEnabled: true, 
			controlIconsEnabled: true, 
			zoomEnabled: true, 
			dblClickZoomEnabled: false, 
			mouseWheelZoomEnabled: true, 
			preventMouseEventsDefault: false, 
			zoomScaleSensitivity: 0.2, 
			minZoom: 0.2, 
			maxZoom: 10, 
			fit: false, 
			contain: false, 
			center: false, 
			refreshRate: "auto",
		});

		editor.screenToWorld = function(pos) {
			var rect = this.svg.getBoundingClientRect();
			var pan = this.panZoom.getPan();
			var zoom = this.panZoom.getZoom();

			return { 
				x: (((pos.x - rect.left) - pan.x) / zoom), 
				y: (((pos.y - rect.top) - pan.y) / zoom)
			};
		};

		window.addEventListener("resize", (e) => {
			editor.panZoom.resize();
		}, true);

		// Load node functions from nodes.js
