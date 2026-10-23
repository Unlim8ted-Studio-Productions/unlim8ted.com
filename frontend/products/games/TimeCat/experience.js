const hero=document.querySelector('.chrono-hero'),portal=document.querySelector('.chronoscope'),canvas=document.getElementById('chrono-canvas'),ctx=canvas.getContext('2d');
const reduced=matchMedia('(prefers-reduced-motion:reduce)').matches;
let width=0,height=0,dpr=1,px=0,py=0,mx=0,my=0,visible=true,running=false;
const stars=Array.from({length:95},(_,i)=>({x:((Math.sin(i*154.27)*43758.54)%1+1)%1,y:((Math.cos(i*74.83)*17654.17)%1+1)%1,size:.3+(i%5)*.15,phase:i*2.17}));
function resize(){width=hero.clientWidth;height=hero.clientHeight;dpr=Math.min(devicePixelRatio||1,1.5);canvas.width=width*dpr;canvas.height=height*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);if(reduced)draw(0);}
function draw(time){ctx.clearRect(0,0,width,height);px+=(mx-px)*.04;py+=(my-py)*.04;hero.style.setProperty('--pointer-x',px.toFixed(3));hero.style.setProperty('--pointer-y',py.toFixed(3));
 for(const s of stars){const x=s.x*width+px*(6+s.size*12),y=s.y*height+py*(3+s.size*8);ctx.globalAlpha=.2+.4*(Math.sin(time*.0003+s.phase)*.5+.5);ctx.fillStyle=s.phase%3>1?'#e5bd78':'#9bacbf';ctx.beginPath();ctx.arc(x,y,s.size,0,Math.PI*2);ctx.fill();}
 const cx=width*.75,cy=height*.49,r=Math.min(width*.38,height*.47);ctx.lineWidth=.6;ctx.globalAlpha=.2;
 for(let orbit=0;orbit<3;orbit++){ctx.beginPath();const pitch=.35+orbit*.43+py*.09,rotation=orbit*.8+time*.000009;
  for(let i=0;i<=150;i++){const a=i/150*Math.PI*2,x=Math.cos(a)*r,y=Math.sin(a)*r*Math.cos(pitch),xx=x*Math.cos(rotation)-y*Math.sin(rotation),yy=x*Math.sin(rotation)+y*Math.cos(rotation);if(i===0)ctx.moveTo(cx+xx,cy+yy);else ctx.lineTo(cx+xx,cy+yy);}ctx.strokeStyle=orbit===1?'#e5bd78':'#728aa6';ctx.stroke();}
 ctx.globalAlpha=1;}
function frame(time){if(!visible||document.hidden||reduced){running=false;return;}draw(time);requestAnimationFrame(frame);}
function start(){if(!running&&!reduced&&visible&&!document.hidden){running=true;requestAnimationFrame(frame);}}
hero.addEventListener('pointermove',e=>{const r=hero.getBoundingClientRect();mx=(e.clientX-r.left)/r.width*2-1;my=(e.clientY-r.top)/r.height*2-1;});hero.addEventListener('pointerleave',()=>{mx=my=0;});
new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;if(visible)start();},{threshold:0}).observe(hero);document.addEventListener('visibilitychange',start);window.addEventListener('resize',resize);resize();start();
for(const button of document.querySelectorAll('[data-era]'))button.addEventListener('click',()=>{const present=button.dataset.era==='present';portal.style.setProperty('--era',present?1:0);document.getElementById('era-readout').textContent='TEMPORAL STATE / '+(present?'PRESENT':'PAST');for(const other of document.querySelectorAll('[data-era]')){const active=other===button;other.classList.toggle('active',active);other.setAttribute('aria-pressed',active);}});
const reveals=document.querySelectorAll('.reveal');if('IntersectionObserver' in window){document.documentElement.classList.add('enhanced');const observer=new IntersectionObserver(entries=>{for(const entry of entries)if(entry.isIntersecting){entry.target.classList.add('seen');observer.unobserve(entry.target);}},{threshold:.1});reveals.forEach(element=>observer.observe(element));}
let scrollPending=false;function progress(){const max=document.documentElement.scrollHeight-innerHeight;document.documentElement.style.setProperty('--page-progress',Math.max(0,Math.min(1,scrollY/Math.max(1,max))));scrollPending=false;}addEventListener('scroll',()=>{if(!scrollPending){scrollPending=true;requestAnimationFrame(progress);}},{passive:true});progress();
