const c=document.getElementById('c'),x=c.getContext('2d'),D=Math.min(2,devicePixelRatio||1);
let W,H,CX,CY,R,P=[],T=[],s=0,v=.14,N=520,last=performance.now();

const pt=u=>{ // true vertical "8": top loop then bottom loop
  let a,y;
  if(u<.5){ a=-Math.PI/2+u*4*Math.PI; y=CY-R }
  else{ a=-Math.PI/2+(u-.5)*4*Math.PI; y=CY+R }
  return {x:CX+R*Math.cos(a),y:y+R*Math.sin(a)};
};

function rs(){
  c.width=innerWidth*D;c.height=innerHeight*D;
  W=c.width;H=c.height;CX=W/2;CY=H/2;R=Math.min(W,H)*.18;
  P=[...Array(N)].map((_,i)=>pt(i/N));
}
addEventListener('resize',rs);rs();

(function loop(t){
  let dt=Math.min(.03,(t-last)/1e3); last=t;
  let i=(s*N)|0,a=P[i],b=P[(i+1)%N],tx=b.x-a.x,ty=b.y-a.y,l=Math.hypot(tx,ty)||1;
  v+= (ty/l)*.18*dt;              // gravity along path
  v=Math.max(.06,Math.min(.35,v));
  s=(s+v*dt)%1;

  let p=P[(s*N)|0]; T.push(p); if(T.length>850)T.shift();

  x.fillStyle='#000';x.fillRect(0,0,W,H);
  x.globalCompositeOperation='lighter';

  if(T.length>1){
    x.beginPath();x.moveTo(T[0].x,T[0].y);
    T.forEach(q=>x.lineTo(q.x,q.y));
    for(let k=3;k;k--){
      x.strokeStyle=`rgba(120,180,255,${.05*k})`;
      x.lineWidth=(2+7*k)*D;x.stroke();
    }
  }

  for(let k=4;k;k--){
    x.beginPath();x.arc(p.x,p.y,7*D+9*k,0,7);
    x.fillStyle=`rgba(160,220,255,${.06*k})`;x.fill();
  }
  x.beginPath();x.arc(p.x,p.y,7*D,0,7);x.fillStyle='#fff';x.fill();

  requestAnimationFrame(loop);
})(last);
