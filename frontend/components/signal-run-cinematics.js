(() => {
  'use strict';
  const clamp = v => Math.max(0, Math.min(1, v));
  const forestCache = new Map();
  const hash = (x,y,seed) => { let n=Math.imul(x,374761393)^Math.imul(y,668265263)^seed;n=Math.imul(n^(n>>>13),1274126177);return ((n^(n>>>16))>>>0)/4294967296 };
  function weather(c,w,h,time,reduced,view={x:0,y:0},surface=null) {
    c.save();
    const t=reduced?0:time;
    // Stable world cells: moving the camera translates existing drops, never reseeds them.
    for(let layer=0;layer<2;layer++) {
      const depth=layer?1:.55,cx=view.x*depth,cy=view.y*depth,wind=85+layer*45,fall=390+layer*160;
      const cw=110,ch=155,x0=Math.floor((cx+t*wind-30)/cw),x1=Math.ceil((cx+w+t*wind+30)/cw);
      const y0=Math.floor((cy-t*fall-50)/ch),y1=Math.ceil((cy+h-t*fall+30)/ch);
      c.strokeStyle=layer?'#c5e2e43b':'#a6c4d224';c.lineWidth=layer?1.1:.7;c.beginPath();
      for(let ix=x0;ix<=x1;ix++)for(let iy=y0;iy<=y1;iy++) {
        const wx=ix*cw+hash(ix,iy,layer+11)*cw-t*wind,wy=iy*ch+hash(ix,iy,layer+81)*ch+t*fall;
        const x=wx-cx,y=wy-cy;
        const floor=surface?surface(view.x+x)-view.y:Infinity;
        if(y>=floor)continue;
        c.moveTo(x,y);c.lineTo(x-10-layer*3,Math.min(y+23+layer*8,floor));
      }
      c.stroke();
    }
    const cw=280,ch=230,cx=view.x,cy=view.y,drift=t*95;
    for(let ix=Math.floor((cx+drift-40)/cw);ix<=Math.ceil((cx+w+drift+40)/cw);ix++)for(let iy=Math.floor((cy-40)/ch);iy<=Math.ceil((cy+h+40)/ch);iy++) {
      const seed=hash(ix,iy,101),wx=ix*cw+seed*cw-drift,wy=iy*ch+hash(ix,iy,309)*ch+Math.sin(t*1.5+seed*6)*14;
      const x=wx-cx,y=wy-cy,a=reduced?seed*6:t*(1+seed)+seed*6,dx=Math.cos(a)*6,dy=Math.sin(a)*6;
      if(surface&&y>surface(view.x+x)-view.y-8)continue;
      c.fillStyle=seed>.3?'#718a6e77':'#a3937555';c.beginPath();c.moveTo(x-dx,y-dy);c.lineTo(x+dy*.5,y-dx*.5);c.lineTo(x+dx,y+dy);c.lineTo(x-dy*.4,y+dx*.4);c.fill();
    }
    const phase=time%14;
    if(!reduced&&phase>6&&phase<6.9) {
      const intensity=Math.sin((phase-6)/.9*Math.PI),x=[540,2250,3450][Math.floor(time/14)%3]-view.x*.55,y=-view.y*.3;
      c.fillStyle=`rgba(189,211,229,${intensity*.08})`;c.fillRect(0,0,w,h);
      c.strokeStyle=`rgba(206,222,237,${intensity*.65})`;c.lineWidth=1.5;c.beginPath();c.moveTo(x,y);c.lineTo(x-14,y+45);c.lineTo(x+3,y+64);c.lineTo(x-23,y+114);c.lineTo(x-10,y+127);c.lineTo(x-38,y+189);c.stroke();
    }
    c.restore();
  }

  function render(c, width, height, options) {
    const { shot, progress, time, reduced } = options;
    const scale = height / 720, w = width / scale, h = 720;
    const t = reduced ? 0 : time, q = reduced ? .5 : clamp(progress);
    c.save(); c.scale(scale, scale);
    const rect = (x,y,ww,hh,color) => { c.fillStyle=color;c.fillRect(x,y,ww,hh); };
    const line = (x,y,xx,yy,color,size=1) => { c.strokeStyle=color;c.lineWidth=size;c.beginPath();c.moveTo(x,y);c.lineTo(xx,yy);c.stroke(); };
    const ellipse = (x,y,rx,ry,color) => { c.fillStyle=color;c.beginPath();c.ellipse(x,y,rx,ry,0,0,Math.PI*2);c.fill(); };
    const sky=c.createLinearGradient(0,0,0,h);sky.addColorStop(0,'#061321');sky.addColorStop(.65,'#294858');sky.addColorStop(1,'#10212d');rect(0,0,w,h,sky);
    function paintForest(offset=0,fw=w) {
      for(let layer=0;layer<4;layer++) {
        c.fillStyle=['#344c5c','#233e4c','#15323e','#0c2530'][layer];c.beginPath();c.moveTo(-30,h);
        for(let x=-30;x<fw+40;x+=15)c.lineTo(x,280+layer*65+Math.sin((x+offset*(layer+1)*.2)*.005+layer*2)*50+Math.sin(x*.015+layer)*16);
        c.lineTo(fw+40,h);c.fill();
        for(let i=-1;i<fw/90+2;i++) {
          const x=i*90-(offset*(layer+1)*.1)%90,y=430+layer*48+Math.sin(i*2)*22,size=.7+layer*.25;
          rect(x-2,y-85*size,4,90*size,['#28434c','#1b3944','#122c38','#09212b'][layer]);
          for(let k=0;k<3;k++){c.fillStyle=['#28434c','#1b3944','#122c38','#09212b'][layer];c.beginPath();c.moveTo(x,y-(140-k*26)*size);c.lineTo(x-(26+k*6)*size,y-(62-k*20)*size);c.lineTo(x+(26+k*6)*size,y-(62-k*20)*size);c.fill()}
        }
      }
    }
    function forest(offset=0) {
      const key=Math.ceil(w);
      if(!forestCache.has(key)) {
        if(forestCache.size>=3)forestCache.clear();
        const image=document.createElement('canvas');image.width=key+240;image.height=720;
        const previous=c;c=image.getContext('2d');paintForest(0,image.width);c=previous;forestCache.set(key,image);
      }
      c.drawImage(forestCache.get(key),-offset,0);
    }
    function figure(x,y,size,color,walking=false,facing=1) {
      c.save();c.translate(x,y);c.scale(size*facing,size);
      const leg=walking?Math.sin(t*11)*7:0;
      line(-5,-12,-8+leg,0,'#a7b9ac',5);line(5,-12,7-leg,0,'#a7b9ac',5);
      rect(-11,-34,22,22,color);ellipse(0,-47,11,13,'#d2c1a1');rect(-9,-58,20,6,'#293d40');
      c.fillStyle=color;c.beginPath();c.moveTo(-8,-35);c.lineTo(-40,-31+Math.sin(t*8)*5);c.lineTo(-26,-24);c.lineTo(5,-31);c.fill();c.restore();
    }
    function helicopter(x,y,size) {
      c.save();c.translate(x,y);c.scale(size,size);c.rotate(Math.sin(t*2)*.035);
      const beam=c.createLinearGradient(0,15,0,300);beam.addColorStop(0,'#e4e9ca33');beam.addColorStop(1,'#e4e9ca00');
      c.fillStyle=beam;c.beginPath();c.moveTo(10,15);c.lineTo(-85+Math.sin(t)*45,300);c.lineTo(115+Math.sin(t)*45,300);c.fill();
      ellipse(0,0,56,25,'#647f84');ellipse(24,-3,25,17,'#183c50');line(-42,-1,-119,-22,'#687f81',12);line(-115,-36,-115,-8,'#a3b6ac',3);
      line(-28,27,37,27,'#a2b6ac',4);line(-20,15,-18,27,'#9caea6',3);line(24,15,25,27,'#9caea6',3);line(-3,-25,-3,-39,'#a2b4a9',5);
      line(-3-Math.cos(t*27)*106,-39,-3+Math.cos(t*27)*106,-39,'#b4c8bf',3);ellipse(50,4,3,3,'#f2c98c');ellipse(-55,-5,3,3,'#d78675');c.restore();
    }
    function face(character) {
      const crew=character==='crew';
      const size=crew?Math.min(w*.32,148):Math.min(w*.43,190),cx=w*.5,cy=crew?355:340;
      c.save();c.translate(cx,cy);c.scale(size/150*(1+q*.035),size/150*(1+q*.035));
      c.translate(0,Math.sin(t*1.3)*1.5);
      // Shoulders, wet hood and scarf are continuous with the explorer's costume.
      c.fillStyle=crew?'#344d59':'#85543c';c.beginPath();c.moveTo(-190,320);c.quadraticCurveTo(-170,145,-75,120);c.quadraticCurveTo(0,90,75,120);c.quadraticCurveTo(170,145,190,320);c.fill();
      ellipse(0,0,122,151,crew?'#6c8288':'#684d3b');
      const skin=c.createLinearGradient(-95,-80,105,100);skin.addColorStop(0,'#e0c9a3');skin.addColorStop(.55,'#b69878');skin.addColorStop(1,'#665b52');
      ellipse(0,10,91,115,skin);ellipse(-91,15,12,24,'#b79a7b');ellipse(91,15,12,24,'#887863');
      {
        c.fillStyle='#243333';c.beginPath();c.moveTo(-90,-5);c.lineTo(-90,-74);c.quadraticCurveTo(0,-145,92,-70);c.lineTo(86,0);c.lineTo(59,-47);c.lineTo(30,-31);c.lineTo(9,-60);c.lineTo(-16,-38);c.lineTo(-43,-49);c.lineTo(-66,-13);c.fill();
      }
      // Eye direction and brows carry the emotion; the mouth moves while speaking.
      const blink=!reduced && Math.sin(t*1.1)>.995, eyeY=10;
      for(const side of [-1,1]) {
        const x=side*36;ellipse(x,eyeY,20,blink?2:8,'#d9d4bd');ellipse(x+(crew?-3:3),eyeY,6,blink?1:7,'#2f4545');ellipse(x+(crew?-1:5),eyeY-2,2,blink?1:2,'#d5e4d6');
        line(x-18,eyeY-18+(crew?0:side*2),x+17,eyeY-20-(crew?0:side*2),'#3a3933',5);
      }
      line(2,22,-7,52,'#8e7862',3);line(-7,52,7,54,'#9e8267',2);
      ellipse(0,83,18,3+(reduced?0:Math.abs(Math.sin(t*12))*3),'#59473f');line(-12,92,12,92,'#b7977a',2);
      if(crew) {
        c.fillStyle='#789096';c.beginPath();c.moveTo(-112,-18);c.lineTo(-100,-100);c.quadraticCurveTo(0,-170,100,-100);c.lineTo(112,-18);c.lineTo(78,-40);c.lineTo(65,-83);c.lineTo(-65,-83);c.lineTo(-78,-40);c.fill();
        rect(-114,-12,22,70,'#213e49');rect(92,-12,22,70,'#213e49');line(102,44,64,83,'#9cb2b1',6);ellipse(54,83,14,7,'#192e38');
      } else { rect(-92,15,14,115,'#243333');rect(80,15,13,115,'#243333'); }
      c.fillStyle=crew?'#54717a':'#bb7952';c.beginPath();c.moveTo(-95,116);c.lineTo(0,148);c.lineTo(95,116);c.lineTo(79,165);c.lineTo(5,185);c.lineTo(-92,153);c.fill();
      for(let i=0;i<12;i++){const x=-105+i*19,y=-70+(i*31)%180;line(x,y,x-3,y+13,'#d1e0da22')}
      c.restore();
    }
    if(shot==='hike'||shot==='collapse') {
      // One hiker stays on the right bank. Her parked car stays on the left.
      forest(shot==='hike'?q*25:25);
      const ground=565;
      rect(0,ground,w,h-ground,'#102c36');
      c.fillStyle='#071723';c.beginPath();c.moveTo(w*.43,ground);c.lineTo(w*.69,ground);c.lineTo(w*.75,h);c.lineTo(w*.37,h);c.fill();
      const carX=w*.18,carW=Math.min(w*.27,140),carY=ground-14;
      rect(carX-carW*.5,carY-carW*.25,carW,carW*.25,'#66858b');
      c.fillStyle='#708d90';c.beginPath();c.moveTo(carX-carW*.35,carY-carW*.25);c.lineTo(carX-carW*.23,carY-carW*.48);c.lineTo(carX+carW*.23,carY-carW*.48);c.lineTo(carX+carW*.4,carY-carW*.25);c.fill();
      rect(carX-carW*.18,carY-carW*.43,carW*.32,carW*.16,'#203f50');ellipse(carX-carW*.3,carY,carW*.1,carW*.1,'#0b1b25');ellipse(carX+carW*.3,carY,carW*.1,carW*.1,'#0b1b25');rect(carX+carW*.43,carY-carW*.2,carW*.06,carW*.07,'#d8be8d');
      c.save();c.translate(w*.43,ground-5);c.rotate(shot==='collapse'?q*1.15:0);
      rect(0,-7,w*.26,14,'#637464');line(0,-40,w*.26,-40,'#889387',2);
      for(let i=0;i<=6;i++)line(i*w*.26/6,-40,i*w*.26/6,0,'#697d71',3);
      c.restore();
      figure(w*(shot==='hike'?.85-q*.08:.77),ground,Math.min(w/300,1.5),'#c1845d',shot==='hike',-1);
      if(shot==='collapse')for(let i=0;i<7;i++) {
        const x=w*.56+(i-3)*w*.025+q*(i-3)*8,y=ground+q*q*180+i%3*10;
        c.save();c.translate(x,y);c.rotate(q*(i-3));rect(-10,-3,24,6,'#687b69');c.restore();
      }
    } else if(shot==='crew') {
      // The landscape, airframe and restrained passenger move in different frames.
      const bank=reduced?0:Math.sin(t*1.15)*.085+Math.sin(t*2.7)*.025;
      const shakeX=reduced?0:Math.sin(t*23)*2+Math.sin(t*37)*.7;
      const shakeY=reduced?0:Math.sin(t*29)*1.6;
      c.save();c.translate(w*.5,350);c.rotate(-bank*.65);c.scale(1.12,1.12);c.translate(-w*.5,-350+(reduced?0:Math.sin(t*.8)*13));
      forest(35+q*100);weather(c,w,h,time,reduced,{x:35+q*100,y:20});c.restore();

      c.save();c.translate(w*.5+shakeX,355+shakeY);c.rotate(bank);c.translate(-w*.5,-355);
      // A visible roof, window pillars, seat, harness and console establish the cabin.
      rect(-70,-60,w+140,190,'#0b1d28');rect(-70,520,w+140,260,'#102630');
      rect(-60,100,w*.09+60,460,'#152c37');rect(w*.91,100,w*.09+60,460,'#152c37');
      line(w*.14,120,w*.06,535,'#71898b',12);line(w*.86,120,w*.94,535,'#71898b',12);
      line(w*.14,126,w*.86,126,'#5e777c',9);line(w*.52,130,w*.52,480,'#203b46',11);
      const seatW=Math.min(w*.63,270);
      rect(w*.5-seatW*.5,310,seatW,300,'#1c333e');ellipse(w*.5,310,seatW*.5,38,'#203b46');
      // A suspended cable swings independently of the fuselage.
      const cableX=w*.18,swing=reduced?0:Math.sin(t*2.2)*11;
      c.strokeStyle='#8da39b';c.lineWidth=2;c.beginPath();c.moveTo(cableX,122);c.quadraticCurveTo(cableX+swing,175,cableX+swing*.7,225);c.stroke();
      rect(cableX+swing*.7-5,221,10,25,'#314f59');
      face('crew');
      const belt=Math.min(w*.22,96);
      line(w*.5-belt,465,w*.5+belt*.65,638,'#13252f',13);line(w*.5+belt,465,w*.5-belt*.65,638,'#13252f',13);
      rect(w*.5-10,573,20,18,'#9caaa2');
      // The attitude indicator rolls against the moving cabin.
      const gaugeW=Math.min(w*.2,88),gx=w*.8,gy=568;
      rect(gx-gaugeW*.5-5,gy-30,gaugeW+10,64,'#071822');
      c.save();c.beginPath();c.rect(gx-gaugeW*.5,gy-25,gaugeW,50);c.clip();c.translate(gx,gy);c.rotate(-bank*2);
      rect(-gaugeW,-50,gaugeW*2,50,'#3e6b7c');rect(-gaugeW,0,gaugeW*2,50,'#746f53');line(-gaugeW,0,gaugeW,0,'#d8dac2',2);c.restore();
      line(gx-12,gy,gx+12,gy,'#edc68b',2);
      for(let i=0;i<4;i++){const x=w*.1+i*w*.07;rect(x,554,w*.045,14,'#334f59');ellipse(x+4,585,2.5,2.5,i%2?'#87b79b':'#c2a473')}
      c.restore();
    } else if(shot==='helicopter') {
      forest(q*25);helicopter(w*(.4+q*.15),240,Math.min(w/250,1.7));
    } else if(shot==='call'&&q<.22) {
      forest(25);
      const ph=260,pw=136,x=w*.5-pw*.5,y=220;
      ellipse(w*.5,y+ph,98,90,'#9f886f');rect(x-8,y-9,pw+16,ph+18,'#172d38');rect(x,y,pw,ph,'#274959');
      c.textAlign='center';c.fillStyle='#dce8df';c.font='13px Arial';c.fillText('Emergency call',w*.5,y+54);c.font='bold 44px Arial';c.fillText('911',w*.5,y+121);c.font='12px Arial';c.fillText('Calling...',w*.5,y+158);ellipse(w*.5,y+214,23,23,'#af7468');
    } else {
      forest(q*8);face('hiker');
    }
    if(shot!=='crew')weather(c,w,h,time,reduced,{x:shot==='hike'?q*25:25,y:0});
    // A lens vignette, restrained push-in and dissolves connect the shots.
    const vignette=c.createRadialGradient(w*.5,320,60,w*.5,350,Math.max(w*.65,450));vignette.addColorStop(0,'#06111c00');vignette.addColorStop(1,'#06111cbb');rect(0,0,w,h,vignette);
    if(!reduced){const fade=Math.max(1-clamp(progress*8),clamp((progress-.9)*10));if(fade>0)rect(0,0,w,h,`rgba(4,12,20,${fade})`)}
    rect(0,0,w,64,'#050e17');rect(0,656,w,64,'#050e17');c.restore();
  }
  window.SignalRunCinema = { render, weather };
})();
