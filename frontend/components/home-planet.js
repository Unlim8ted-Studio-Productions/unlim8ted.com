(() => {
      if (typeof THREE === "undefined") { window.homePlanetAvailable = false; document.getElementById("planetIntro").hidden = true; return; }
      const isMobile = window.matchMedia("(max-width: 900px)").matches;
      const canvas = document.getElementById("homePlanetCanvas");
      let renderer;
      try { renderer = new THREE.WebGLRenderer({ canvas, antialias: false, alpha: true, powerPreference: "high-performance" }); }
      catch { window.homePlanetAvailable = false; document.getElementById("planetIntro").hidden = true; return; }
      window.homePlanetAvailable = true;
      let renderedWidth = 0, renderedHeight = 0, renderedPixelRatio = 0;
      function sizeToPreview() {
        const width = Math.max(1, canvas.clientWidth);
        const height = Math.max(1, canvas.clientHeight);
        const displayedHeight = canvas.getBoundingClientRect().height;
        const zoom = Math.max(1, displayedHeight / height);
        const budget=isMobile?450000:850000;
        const pixelRatio = Math.min(isMobile?1.25:1.5,devicePixelRatio*zoom,Math.sqrt(budget/(width*height)));
        if (width === renderedWidth && height === renderedHeight && Math.abs(pixelRatio - renderedPixelRatio) < .15) return;
        renderedWidth = width;
        renderedHeight = height;
        renderedPixelRatio = pixelRatio;
        renderer.setPixelRatio(pixelRatio);
        renderer.setSize(width, height, false);
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
      }

      const scene = new THREE.Scene();
      scene.fog = new THREE.Fog(0x050505, 9, 18);

      const camera = new THREE.PerspectiveCamera(50, 1, .1, 1000);
      camera.position.set(0, .2, 9.4);

      const group = new THREE.Group();
      scene.add(group);

      const world = new THREE.Group();
      group.add(world);

      const earth = new THREE.Mesh(
        new THREE.SphereGeometry(1.72, isMobile ? 24 : 32, isMobile ? 24 : 32),
        new THREE.MeshToonMaterial({
          color: 0x2f78d2,
          gradientMap: null
        })
      );
      world.add(earth);

      const earthGlow = new THREE.Mesh(
        new THREE.SphereGeometry(1.88, 32, 32),
        new THREE.MeshBasicMaterial({
          color: 0x245f95,
          transparent: true,
          opacity: .08
        })
      );
      world.add(earthGlow);

      const atmosphere = new THREE.Mesh(
        new THREE.TorusGeometry(2.15, .06, 18, 100),
        new THREE.MeshBasicMaterial({
          color: 0xffffff,
          transparent: true,
          opacity: .1
        })
      );
      atmosphere.rotation.x = Math.PI * .34;
      atmosphere.rotation.y = -.45;
      group.add(atmosphere);

      function addLand(x, y, z, rx, ry, rz, sx, sy, sz, color = 0x67d66d) {
        const land = new THREE.Mesh(
          new THREE.SphereGeometry(.32, 16, 16),
          new THREE.MeshToonMaterial({ color })
        );
        land.position.set(x, y, z);
        land.rotation.set(rx, ry, rz);
        land.scale.set(sx, sy, sz);
        earth.add(land);
      }

      function addLandAt(lat, lon, sx, sy, sz, color = 0x6cae61) {
        const phi = THREE.MathUtils.degToRad(90 - lat);
        const theta = THREE.MathUtils.degToRad(lon);
        const radius = 1.78;
        const x = radius * Math.sin(phi) * Math.cos(theta);
        const y = radius * Math.cos(phi);
        const z = radius * Math.sin(phi) * Math.sin(theta);

        const land = new THREE.Mesh(
          new THREE.SphereGeometry(.34, 16, 16),
          new THREE.MeshToonMaterial({ color })
        );

        land.position.set(x, y, z);
        land.lookAt(0, 0, 0);
        land.rotateY(Math.PI / 2);
        land.scale.set(sx, sy, Math.max(.34, sz * 2.25));
        earth.add(land);
      }

      [
        [48, -105, 1.55, .78, .28, 0x6cae61],
        [38, -98, 1.05, .58, .22, 0x5f9f56],
        [19, -99, .72, .34, .18, 0x75ba69],
        [-12, -60, .88, 1.16, .22, 0x5d9853],
        [-28, -62, .52, .88, .16, 0x6eaf61],
        [72, -42, .56, .26, .12, 0x83c977],
        [55, 15, .76, .42, .18, 0x739f5c],
        [13, 20, 1.86, 1.28, .32, 0x6ba65f],
        [1, 30, 1.02, .86, .24, 0x588d4f],
        [-22, 24, .78, 1.02, .22, 0x77ba6a],
        [23, 45, .56, .26, .12, 0x90cf7d],
        [48, 85, 1.92, .84, .3, 0x6daa61],
        [62, 96, 1.18, .44, .18, 0x5d8c50],
        [22, 78, .72, .78, .18, 0x8bca78],
        [15, 105, 1.08, .6, .24, 0x73b666],
        [-3, 118, .92, .48, .18, 0x5a914f],
        [-25, 135, 1.12, .7, .22, 0x6eab60],
        [-41, 146, .44, .26, .12, 0x86c774],
        [36, 138, .38, .22, .1, 0x93d381],
        [64, -18, .3, .16, .08, 0xa2de8d]
      ].forEach(args => addLandAt(...args));

      const cloudBands = [];

      for (let i = 0; i < 3; i++) {
        const cloud = new THREE.Mesh(
          new THREE.TorusGeometry(1.94 + i * .04, .03, 14, 72),
          new THREE.MeshBasicMaterial({
            color: 0xffffff,
            transparent: true,
            opacity: .22 - i * .04
          })
        );
        cloud.rotation.set(Math.PI * (.2 + i * .14), i * .85, i * .3);
        world.add(cloud);
        cloudBands.push(cloud);
      }

      const moonlet = new THREE.Mesh(
        new THREE.SphereGeometry(.2, 18, 18),
        new THREE.MeshToonMaterial({ color: 0xffefae })
      );
      moonlet.position.set(-2.3, 1.2, -.4);
      group.add(moonlet);

      const orbiters = [];

      function createFilmSlate() {
        const item = new THREE.Group();
        const body = new THREE.Mesh(
          new THREE.BoxGeometry(.7, .52, .18),
          new THREE.MeshToonMaterial({ color: 0x171717 })
        );
        item.add(body);

        const clap = new THREE.Mesh(
          new THREE.BoxGeometry(.8, .16, .18),
          new THREE.MeshToonMaterial({ color: 0xf2f2f2 })
        );
        clap.position.y = .34;
        clap.rotation.z = -.2;
        item.add(clap);

        for (let i = 0; i < 4; i++) {
          const stripe = new THREE.Mesh(
            new THREE.BoxGeometry(.1, .16, .19),
            new THREE.MeshToonMaterial({ color: i % 2 ? 0x121212 : 0xf5c84c })
          );
          stripe.position.set(-.3 + i * .2, .34, .01);
          stripe.rotation.z = -.2;
          item.add(stripe);
        }

        const label = new THREE.Mesh(
          new THREE.BoxGeometry(.48, .08, .02),
          new THREE.MeshToonMaterial({ color: 0xffffff })
        );
        label.position.set(0, -.06, .1);
        item.add(label);

        const lineA = new THREE.Mesh(
          new THREE.BoxGeometry(.42, .03, .02),
          new THREE.MeshToonMaterial({ color: 0xff6b6b })
        );
        lineA.position.set(0, .06, .1);
        item.add(lineA);

        const lineB = new THREE.Mesh(
          new THREE.BoxGeometry(.34, .03, .02),
          new THREE.MeshToonMaterial({ color: 0x58c5ff })
        );
        lineB.position.set(-.04, -.16, .1);
        item.add(lineB);

        return item;
      }

      function createController() {
        const item = new THREE.Group();
        const shell = new THREE.Mesh(
          new THREE.BoxGeometry(.96, .44, .18),
          new THREE.MeshToonMaterial({ color: 0xf3f3f1 })
        );
        shell.scale.set(1, .92, 1);
        item.add(shell);



        const inset = new THREE.Mesh(
          new THREE.BoxGeometry(.64, .18, .03),
          new THREE.MeshToonMaterial({ color: 0xd8dad7 })
        );
        inset.position.set(0, .02, .09);
        item.add(inset);

        const dpadH = new THREE.Mesh(
          new THREE.BoxGeometry(.22, .06, .08),
          new THREE.MeshToonMaterial({ color: 0x202020 })
        );
        dpadH.position.set(-.23, .03, .12);
        item.add(dpadH);

        const dpadV = dpadH.clone();
        dpadV.rotation.z = Math.PI / 2;
        item.add(dpadV);

        [-.12, .12].forEach(x => {
          const stickBase = new THREE.Mesh(
            new THREE.CylinderGeometry(.075, .075, .035, 20),
            new THREE.MeshToonMaterial({ color: 0x1f1f1f })
          );
          stickBase.rotation.x = Math.PI / 2;
          stickBase.position.set(x, -.11, .11);
          item.add(stickBase);
        });

        const startBar = new THREE.Mesh(
          new THREE.BoxGeometry(.16, .04, .05),
          new THREE.MeshToonMaterial({ color: 0x141414 })
        );
        startBar.position.set(0, .03, .12);
        item.add(startBar);

        ["#ff6b6b", "#ffd93d", "#6bff95", "#58c5ff"].forEach((color, i) => {
          const btn = new THREE.Mesh(
            new THREE.SphereGeometry(.052, 12, 12),
            new THREE.MeshToonMaterial({ color })
          );
          btn.position.set(.22 + (i % 2) * .1, .08 - (i > 1 ? .1 : 0), .12);
          item.add(btn);
        });

        return item;
      }

      function createHeadphones() {
        const item = new THREE.Group();
        const bandPath = [
          new THREE.Vector3(-.34, .3, 0),
          new THREE.Vector3(-.28, .74, 0),
          new THREE.Vector3(.02, .98, 0),
          new THREE.Vector3(.26, .94, 0),
          new THREE.Vector3(.45, .64, 0),
          new THREE.Vector3(.45, .14, 0)
        ];

        function addBandSegment(a, b, color, thickness = .13, depth = .12) {
          const dx = b.x - a.x;
          const dy = b.y - a.y;
          const len = Math.hypot(dx, dy);
          const segment = new THREE.Mesh(
            new THREE.BoxGeometry(thickness, len, depth),
            new THREE.MeshToonMaterial({ color })
          );
          segment.position.set((a.x + b.x) / 2, (a.y + b.y) / 2, 0);
          segment.rotation.z = Math.atan2(dy, dx) - Math.PI / 2;
          item.add(segment);
          return segment;
        }

        for (let i = 0; i < bandPath.length - 1; i++) {
          const color = i < 2 ? 0xffffff : 0x181818;
          addBandSegment(bandPath[i], bandPath[i + 1], color, i < 2 ? .14 : .13, .12);
        }

        const innerBandPath = [
          new THREE.Vector3(-.18, .34, .01),
          new THREE.Vector3(-.14, .64, .01),
          new THREE.Vector3(.06, .82, .01),
          new THREE.Vector3(.22, .79, .01),
          new THREE.Vector3(.34, .58, .01),
          new THREE.Vector3(.34, .2, .01)
        ];

        for (let i = 0; i < innerBandPath.length - 1; i++) {
          addBandSegment(innerBandPath[i], innerBandPath[i + 1], 0x313131, .08, .08);
        }

        function createEarCup(side, isFront) {
          const cup = new THREE.Group();

          const shell = new THREE.Mesh(
            new THREE.CylinderGeometry(.22, .24, .16, 8),
            new THREE.MeshToonMaterial({ color: isFront ? 0xffffff : 0x1a1a1a })
          );
          shell.rotation.z = Math.PI / 2;
          shell.scale.set(1, .95, 1);
          cup.add(shell);

          const rim = new THREE.Mesh(
            new THREE.CylinderGeometry(.16, .18, .12, 8),
            new THREE.MeshToonMaterial({ color: 0x2a2a2a })
          );
          rim.rotation.z = Math.PI / 2;
          rim.position.x = .02;
          cup.add(rim);

          const pad = new THREE.Mesh(
            new THREE.CylinderGeometry(.1, .12, .08, 8),
            new THREE.MeshToonMaterial({ color: 0xf4f4f4 })
          );
          pad.rotation.z = Math.PI / 2;
          pad.position.x = .08;
          cup.add(pad);

          if (isFront) {
            const accentBase = new THREE.Mesh(
              new THREE.BoxGeometry(.13, .18, .03),
              new THREE.MeshToonMaterial({ color: 0x454545 })
            );
            accentBase.position.set(-.1, -.02, .09);
            cup.add(accentBase);

            const redBarA = new THREE.Mesh(
              new THREE.BoxGeometry(.03, .12, .02),
              new THREE.MeshToonMaterial({ color: 0xe34633 })
            );
            redBarA.position.set(-.12, -.01, .11);
            redBarA.rotation.z = .28;
            cup.add(redBarA);

            const redBarB = new THREE.Mesh(
              new THREE.BoxGeometry(.07, .03, .02),
              new THREE.MeshToonMaterial({ color: 0xe34633 })
            );
            redBarB.position.set(-.09, .04, .11);
            redBarB.rotation.z = -.42;
            cup.add(redBarB);
          }

          cup.position.set(side * .38, -.18, side < 0 ? .02 : -.04);
          cup.rotation.z = side < 0 ? .08 : -.08;
          cup.rotation.y = side < 0 ? .18 : -.28;
          return cup;
        }

        const leftConnector = new THREE.Mesh(
          new THREE.BoxGeometry(.12, .14, .12),
          new THREE.MeshToonMaterial({ color: 0xbdbdbd })
        );
        leftConnector.position.set(-.34, -.02, 0);
        leftConnector.rotation.z = -.42;
        item.add(leftConnector);

        const rightConnector = new THREE.Mesh(
          new THREE.BoxGeometry(.12, .2, .12),
          new THREE.MeshToonMaterial({ color: 0x1d1d1d })
        );
        rightConnector.position.set(.42, .02, -.01);
        rightConnector.rotation.z = .12;
        item.add(rightConnector);

        const leftCup = createEarCup(-1, true);
        item.add(leftCup);

        const rightCup = createEarCup(1, false);
        rightCup.scale.set(1.14, 1.14, 1.14);
        rightCup.position.set(.44, -.15, -.06);
        item.add(rightCup);

        item.rotation.z = -.12;
        item.scale.set(.92, .92, .92);

        return item;
      }

      function createStar() {
        const shape = new THREE.Shape();
        for (let i = 0; i < 10; i++) {
          const angle = (i / 10) * Math.PI * 2 - Math.PI / 2;
          const radius = i % 2 === 0 ? .42 : .18;
          const x = Math.cos(angle) * radius;
          const y = Math.sin(angle) * radius;
          if (i === 0) shape.moveTo(x, y);
          else shape.lineTo(x, y);
        }
        shape.closePath();
        return new THREE.Mesh(
          new THREE.ExtrudeGeometry(shape, { depth: .12, bevelEnabled: false }),
          new THREE.MeshToonMaterial({ color: 0xffcf52 })
        );
      }

      function createCamera() {
        const item = new THREE.Group();
        const body = new THREE.Mesh(
          new THREE.BoxGeometry(.76, .46, .24),
          new THREE.MeshToonMaterial({ color: 0x1a1a1a })
        );
        item.add(body);

        const top = new THREE.Mesh(
          new THREE.BoxGeometry(.28, .12, .18),
          new THREE.MeshToonMaterial({ color: 0x2b2b2b })
        );
        top.position.set(-.14, .28, 0);
        item.add(top);

        const lensOuter = new THREE.Mesh(
          new THREE.CylinderGeometry(.18, .18, .18, 24),
          new THREE.MeshToonMaterial({ color: 0x4a4f58 })
        );
        lensOuter.rotation.x = Math.PI / 2;
        lensOuter.position.set(.12, 0, .12);
        item.add(lensOuter);

        const lensInner = new THREE.Mesh(
          new THREE.CylinderGeometry(.1, .1, .19, 20),
          new THREE.MeshToonMaterial({ color: 0x6dc5ff })
        );
        lensInner.rotation.x = Math.PI / 2;
        lensInner.position.set(.12, 0, .15);
        item.add(lensInner);

        const flash = new THREE.Mesh(
          new THREE.BoxGeometry(.16, .12, .04),
          new THREE.MeshToonMaterial({ color: 0xffd24c })
        );
        flash.position.set(-.26, .04, .15);
        item.add(flash);

        const shutter = new THREE.Mesh(
          new THREE.SphereGeometry(.045, 12, 12),
          new THREE.MeshToonMaterial({ color: 0xff6b6b })
        );
        shutter.position.set(.24, .18, .14);
        item.add(shutter);

        return item;
      }

      function addOrbiter(factory, options) {
        const pivot = new THREE.Group();
        group.add(pivot);

        const holder = new THREE.Group();
        pivot.add(holder);

        const mesh = factory();
        holder.add(mesh);

        const trails = [];
        for (let i = 0; i < 3; i++) {
          const trail = factory();
          trail.traverse(node => {
            if (node.isMesh) {
              node.material = node.material.clone();
              node.material.transparent = true;
              node.material.opacity = .18 - (i * .045);
            }
          });
          holder.add(trail);
          trails.push(trail);
        }

        orbiters.push({
          pivot,
          holder,
          mesh,
          trails,
          radius: options.radius,
          height: options.height,
          speed: options.speed,
          offset: options.offset,
          tilt: options.tilt,
          wobble: options.wobble,
          scale: options.scale
        });
      }

      addOrbiter(createFilmSlate, { radius: 3.7, height: .3, speed: .95, offset: .2, tilt: .5, wobble: .18, scale: .88 });
      addOrbiter(createController, { radius: 4.15, height: -.2, speed: 1.2, offset: 1.8, tilt: -.35, wobble: .24, scale: .96 });
      addOrbiter(createHeadphones, { radius: 3.85, height: .8, speed: 1.45, offset: 3.1, tilt: .75, wobble: .15, scale: .9 });
      addOrbiter(createStar, { radius: 4.45, height: -.75, speed: 1.65, offset: 4.2, tilt: .22, wobble: .2, scale: .72 });
      addOrbiter(createCamera, { radius: 3.95, height: -.95, speed: 1.05, offset: 5.1, tilt: -.62, wobble: .28, scale: .84 });

      const count = isMobile ? 360 : 900;
      const positions = new Float32Array(count * 3);
      const colors = new Float32Array(count * 3);
      const palette = [0x2b7fd1, 0xd8a935, 0xcf6297, 0x5fa85f, 0xd9e6f2];

      for (let i = 0; i < count; i++) {
        positions[i * 3] = (Math.random() - .5) * 30;
        positions[i * 3 + 1] = (Math.random() - .5) * 18;
        positions[i * 3 + 2] = (Math.random() - .5) * 24;

        const color = new THREE.Color(palette[(Math.random() * palette.length) | 0]);
        colors[i * 3] = color.r;
        colors[i * 3 + 1] = color.g;
        colors[i * 3 + 2] = color.b;
      }

      const pGeo = new THREE.BufferGeometry();
      pGeo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      pGeo.setAttribute("color", new THREE.BufferAttribute(colors, 3));

      const pMat = new THREE.PointsMaterial({
        size: isMobile ? .018 : .024,
        transparent: true,
        opacity: .52,
        depthWrite: false,
        vertexColors: true
      });

      const particles = new THREE.Points(pGeo, pMat);
      scene.add(particles);

      const keyLight = new THREE.DirectionalLight(0xffffff, 1.45);
      keyLight.position.set(4.5, 3.6, 5.2);
      scene.add(keyLight);

      const fillLight = new THREE.PointLight(0xb85f8e, 1.05, 24);
      fillLight.position.set(-4.4, -1.4, 3.6);
      scene.add(fillLight);

      const rimLight = new THREE.PointLight(0x3b8ec9, 1.15, 30);
      rimLight.position.set(0, 2.5, -4.8);
      scene.add(rimLight);

      scene.add(new THREE.AmbientLight(0xffffff, .38));

      let px = 0, py = 0, tx = 0, ty = 0, scrollP = 0, planetVisible=true, planetRaf=0, planetDirty=true, planetReduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
      function queuePlanet(){if(!planetRaf&&!document.hidden&&planetVisible&&(!planetReduced||planetDirty))planetRaf=requestAnimationFrame(render)}
      window.setHomePlanetProgress=(value,visible=true)=>{
        const next=Math.max(0,Math.min(1,value));
        if(next!==scrollP||visible!==planetVisible)planetDirty=true;
        scrollP=next;planetVisible=visible;
        if(visible)queuePlanet();else{cancelAnimationFrame(planetRaf);planetRaf=0}
      };
      window.addEventListener('resize',()=>{planetDirty=true;queuePlanet()},{passive:true});
      document.addEventListener('home-motion-change',event=>{planetReduced=event.detail.reduced;planetDirty=true;queuePlanet()});
      document.addEventListener('visibilitychange',()=>{if(document.hidden){cancelAnimationFrame(planetRaf);planetRaf=0}else queuePlanet()});

      window.addEventListener("pointermove", e => {
        if(planetReduced)return;
        tx = (e.clientX / innerWidth - .5);
        ty = (e.clientY / innerHeight - .5);
      });


      let lastWebglFrame = 0;
      function render(now) {
        planetRaf=0;
        if(document.hidden||!planetVisible||canvas.parentElement.hidden)return;
        if(now-lastWebglFrame<32){queuePlanet();return}
        lastWebglFrame = now;
        sizeToPreview();
        const time = now * .001;

        px += (tx - px) * .04;
        py += (ty - py) * .04;

        group.rotation.y += .0022 + scrollP * .0035;
        group.rotation.x = .08 + py * .08;
        group.position.x = px * .5;
        group.position.y = -py * .28;

        camera.position.x = px * .65;
        camera.position.y = -py * .4 + .18;
        camera.position.z = 9.4 - scrollP * 1.8;
        camera.lookAt(group.position);

        world.rotation.y += .0032;
        world.rotation.z = Math.sin(time * .8) * .06;
        earthGlow.scale.setScalar(1.01 + Math.sin(time * 1.8) * .02);
        atmosphere.rotation.z += .0028;
        moonlet.position.x = -2.3 + Math.cos(time * .9) * .22;
        moonlet.position.y = 1.2 + Math.sin(time * .9) * .16;

        cloudBands.forEach((cloud, i) => {
          cloud.rotation.y += .0015 + (i * .0005);
          cloud.rotation.z = Math.sin(time * (1.2 + i * .25)) * 0.16;
        });

        orbiters.forEach(orbiter => {
          const angle = time * orbiter.speed + orbiter.offset + scrollP * 2.4;
          const x = Math.cos(angle) * orbiter.radius;
          const z = Math.sin(angle) * orbiter.radius * .9;
          const y = orbiter.height + Math.sin(angle * 1.7) * orbiter.wobble;

          orbiter.pivot.rotation.set(orbiter.tilt, angle * .22, 0);
          orbiter.holder.position.set(x, y, z);
          orbiter.holder.scale.setScalar(orbiter.scale + (Math.sin(time * 2.2 + orbiter.offset) * .03));
          orbiter.mesh.rotation.x = time * 1.6;
          orbiter.mesh.rotation.y = time * 1.1 + orbiter.offset;
          orbiter.mesh.rotation.z = Math.sin(time * 2.4 + orbiter.offset) * .25;

          orbiter.trails.forEach((trail, index) => {
            const lag = (index + 1) * .12;
            trail.position.set(
              -Math.cos(angle) * lag * .9,
              Math.sin(angle * 1.7 + index) * lag * .08,
              -Math.sin(angle) * lag * .62
            );
            trail.scale.setScalar(1 - index * .08);
            trail.rotation.copy(orbiter.mesh.rotation);
            trail.rotation.y -= lag * 1.6;
          });
        });

        particles.rotation.y += .0007 + scrollP * .0012;
        particles.rotation.x = py * .04;

        renderer.render(scene, camera);
        planetDirty=false;
        queuePlanet();
      }

      queuePlanet();

    })();
