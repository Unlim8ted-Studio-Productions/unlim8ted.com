import {chooseVariant,reconstruct} from './chunks.js';
const buildBase=new URL('https://assets.unlim8ted.com/games/TimeCat/');
const manifestUrl=new URL('manifest.json',buildBase);
const $=id=>document.getElementById(id),shell=$('game-shell'),canvas=$('unity-canvas');
const state={instance:null,loading:false,loaded:false,touch:/Android|iPhone|iPad|iPod/i.test(navigator.userAgent)||(navigator.maxTouchPoints>1&&matchMedia('(pointer:coarse)').matches),muted:false,paused:false,scene:'Title'};
let manifestPromise=fetch(manifestUrl,{cache:'no-cache'}).then(r=>{if(!r.ok)throw new Error('The game files are not available yet.');return r.json();});
manifestPromise.then(m=>{const mb=Object.values(m.files).reduce((n,f)=>n+chooseVariant(f).bytes,0)/1048576;$('download-note').textContent=`Browser game \u00b7 ${Math.ceil(mb)} MB initial download`}).catch(()=>{});
const send=(method,value='')=>state.instance?.SendMessage('TimeCatWebBridge',method,String(value));
function updateTouch(){
 const playing=state.loaded&&!state.paused,scene=state.scene;
 $('touch-controls').hidden=!(state.touch&&playing&&scene!=='Title');
 $('touch-toggle').setAttribute('aria-pressed',state.touch);send('SetTouchMode',state.touch?'1':'0');
 $('game-pause').hidden=!state.loaded||state.paused||scene==='Title';
}
window.TimeCatScene=name=>{state.scene=name;$('game-state').textContent=name==='Title'?'TIMECAT / THE BEGINNING':name==='Intro'?'FOLLOW THE STORY':`TIMECAT / ${name.replace('_',' ').toUpperCase()}`;updateTouch();};
window.TimeCatPause=paused=>{state.paused=!!paused;updateTouch();if(!paused)canvas.focus({preventScroll:true});};
function setPaused(paused){if(!state.loaded)return;if(state.scene==='Title'&&paused){send('ClearInput');return;}window.TimeCatPause(paused);send('SetPaused',paused?'1':'0');}
async function fullscreen(){
 if(document.fullscreenElement){await document.exitFullscreen();return;}
 if(shell.classList.contains('mobile-expanded')){shell.classList.remove('mobile-expanded');document.body.classList.remove('no-scroll');return;}
 try{if(shell.requestFullscreen){await shell.requestFullscreen();return;}}catch(e){console.info('Using window-sized play mode.');}
 shell.classList.add('mobile-expanded');document.body.classList.add('no-scroll');
}
async function launch(){
 if(state.loaded){setPaused(false);send('SetMuted',state.muted?'1':'0');canvas.focus({preventScroll:true});return;}
 if(state.loading)return;
 state.loading=true;shell.classList.add('loading');$('loading').hidden=false;$('retry').hidden=true;
 $('loading-title').textContent='LOADING TIMECAT';$('progress-text').textContent='Preparing the game...';$('progress-fill').style.width='0%';
 if(state.touch&&!document.fullscreenElement&&!shell.classList.contains('mobile-expanded'))fullscreen();
 const urls=[];
 try{
  if(!document.createElement('canvas').getContext('webgl2',{antialias:false}))throw new Error('This browser cannot create a WebGL 2 context. Try a current browser with hardware acceleration enabled.');
  manifestPromise=fetch(manifestUrl,{cache:'no-cache'}).then(r=>{if(!r.ok)throw new Error('The game files are not available yet.');return r.json();});
  const manifest=await manifestPromise;
  const selections=Object.entries(manifest.files).map(([key,file])=>({key,file,variant:chooseVariant(file)}));
  const total=selections.reduce((n,s)=>n+s.variant.bytes,0);let received=0;
  const config={companyName:'Unlim8ted Studios',productName:'TimeCat',productVersion:manifest.version||'1.0',streamingAssetsUrl:new URL('StreamingAssets/',buildBase).href,devicePixelRatio:Math.min(window.devicePixelRatio||1,state.touch?1.35:2),matchWebGLToCanvasSize:true,showBanner:(message,type)=>{if(type==='error')console.error(message);else console.info(message);}};
  for(const selection of selections){
   const url=await reconstruct(selection.file,selection.variant,{base:buildBase,onBytes:bytes=>{received+=bytes;$('progress-fill').style.width=`${Math.min(82,received/total*82)}%`;$('progress-text').textContent=`Opening the world \u00b7 ${(received/1048576).toFixed(1)} / ${(total/1048576).toFixed(1)} MB`;}});
   config[selection.key]=url;urls.push(url);
  }
  $('loading-title').textContent='LOADING TIMECAT';$('progress-text').textContent='Starting the game...';
  await new Promise((resolve,reject)=>{const script=document.createElement('script');script.src=new URL(manifest.loaderUrl,buildBase).href;script.onload=resolve;script.onerror=()=>reject(new Error('Could not open the game engine. Please retry.'));document.head.appendChild(script);});
  canvas.style.visibility='visible';
  const starting=window.createUnityInstance(canvas,config,progress=>{$('progress-fill').style.width=`${82+progress*18}%`;});
  canvas.focus({preventScroll:true});
  state.instance=await starting;
  window.timeCatInstance=state.instance;state.loaded=true;state.loading=false;
  canvas.classList.add('ready');$('loading').hidden=true;$('game-poster').hidden=true;shell.classList.remove('loading');
  send('SetMuted',state.muted?'1':'0');updateTouch();canvas.focus({preventScroll:true});
 }catch(error){console.error(error);state.loading=false;$('loading-title').textContent='COULD NOT START';$('progress-text').textContent=error.message||'The game could not start. Please try again.';$('retry').hidden=false;manifestPromise=fetch(manifestUrl,{cache:'reload'}).then(r=>{if(!r.ok)throw new Error('The game files could not be reached.');return r.json();});}
 finally{for(const url of urls)URL.revokeObjectURL(url);}
}
window.TimeCatExit=()=>{setPaused(true);if(document.fullscreenElement)document.exitFullscreen().catch(()=>{});shell.classList.remove('mobile-expanded');document.body.classList.remove('no-scroll');$('home').scrollIntoView({behavior:matchMedia('(prefers-reduced-motion:reduce)').matches?'instant':'smooth'});};
$('launch').addEventListener('click',launch);$('retry').addEventListener('click',()=>{if(window.createUnityInstance)location.reload();else launch();});
document.querySelectorAll('[data-play]').forEach(link=>link.addEventListener('click',()=>{setTimeout(launch,180);}));
$('fullscreen').addEventListener('click',fullscreen);
$('sound-toggle').addEventListener('click',()=>{state.muted=!state.muted;send('SetMuted',state.muted?'1':'0');$('sound-toggle').textContent=state.muted?'Sound off':'Sound on';$('sound-toggle').setAttribute('aria-pressed',state.muted);});
$('touch-toggle').addEventListener('click',()=>{state.touch=!state.touch;send('ClearInput');resetStick();updateTouch();});
$('game-pause').addEventListener('click',()=>setPaused(true));
document.addEventListener('visibilitychange',()=>{if(document.hidden&&state.loaded)setPaused(true);});
window.addEventListener('blur',()=>{send('ClearInput');resetStick();if(state.loaded)setPaused(true);});
document.addEventListener('fullscreenchange',()=>{if(state.loaded&&!state.paused)canvas.focus({preventScroll:true});});
const dialog=$('controls-dialog');$('controls-open').addEventListener('click',()=>{if(state.loaded)setPaused(true);dialog.showModal();});$('controls-close').addEventListener('click',()=>dialog.close());dialog.addEventListener('click',event=>{if(event.target===dialog){const r=dialog.getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)dialog.close();}});
const range=$('timeline');function updateReality(){range.closest('.reality').style.setProperty('--reveal',range.value+'%');}if(range){range.addEventListener('input',updateReality);updateReality();}
let tick=false;if(!matchMedia('(prefers-reduced-motion:reduce)').matches)window.addEventListener('scroll',()=>{if(tick)return;tick=true;requestAnimationFrame(()=>{document.documentElement.style.setProperty('--scroll',Math.min(window.scrollY,1000));tick=false;});},{passive:true});
let stickPointer=null;const stick=$('stick'),thumb=$('stick-thumb');
function resetStick(){stickPointer=null;thumb.style.transform='translate(0,0)';send('SetTouchState',JSON.stringify({x:0,y:0}));}
function moveStick(event){if(event.pointerId!==stickPointer)return;const r=stick.getBoundingClientRect(),radius=r.width*.36;let x=(event.clientX-r.left-r.width/2)/radius,y=-(event.clientY-r.top-r.height/2)/radius;const magnitude=Math.hypot(x,y);if(magnitude>1){x/=magnitude;y/=magnitude;}thumb.style.transform=`translate(${x*radius}px,${-y*radius}px)`;send('SetTouchState',JSON.stringify({x,y}));event.preventDefault();}
stick.addEventListener('pointerdown',event=>{stickPointer=event.pointerId;stick.setPointerCapture(event.pointerId);moveStick(event);});stick.addEventListener('pointermove',moveStick);for(const type of ['pointerup','pointercancel','lostpointercapture'])stick.addEventListener(type,event=>{if(event.pointerId===stickPointer)resetStick();});
document.querySelectorAll('[data-action]').forEach(button=>{button.addEventListener('pointerdown',event=>{event.preventDefault();button.setPointerCapture(event.pointerId);button.classList.add('pressed');send('Action',button.dataset.action);});for(const type of ['pointerup','pointercancel','lostpointercapture'])button.addEventListener(type,()=>button.classList.remove('pressed'));});
document.addEventListener('keydown',event=>{if(event.key==='Escape'&&shell.classList.contains('mobile-expanded')){shell.classList.remove('mobile-expanded');document.body.classList.remove('no-scroll');}});
updateTouch();
