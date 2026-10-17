const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const html = fs.readFileSync(path.join(__dirname, '../frontend/index.html'), 'utf8');

function engine() {
  const held = new Set(), elements = {};
  const context = {
    mix: (a,b,t) => a+(b-a)*t,
    clamp: (v,a=0,b=1) => Math.max(a,Math.min(b,v)),
    inputDown: n => held.has(n), $: id => elements[id] ||= {},
    clearInputs: () => held.clear(), wake() {}, dirty: false,
  };
  vm.createContext(context);
  vm.runInContext(html.slice(html.indexOf('      const ground ='),html.indexOf('      function rect(')) + ';globalThis.api={game,gameStep,resetGame,nextStory,ground,groundY,archways};',context);
  return { ...context.api, held };
}
function skip(e) { while(e.game.scene)e.nextStory(); }
function move(e,x) {
  for(let i=0;i<1800;i++) {
    skip(e); e.held.clear(); if(e.game.x>x+5)e.held.add('left'); if(e.game.x<x-5)e.held.add('right');
    e.gameStep(1/120);
    if(e.game.won)return;
    if(Math.abs(e.game.x-x)<7&&Math.abs(e.game.vx)<10) {
      e.held.clear(); for(let j=0;j<100;j++){skip(e);e.gameStep(1/120)} return;
    }
  }
  assert.fail('Could not walk to '+x);
}
function jump(e,x) {
  e.held.add('jump');e.game.jumpBuffer=.14;
  for(let i=0;i<180;i++) {
    skip(e);e.held.delete('left');e.held.delete('right');
    if(e.game.x>x+4)e.held.add('left');if(e.game.x<x-4)e.held.add('right');e.gameStep(1/120);
    if(i>12&&e.game.grounded){e.held.clear();return}
  }
  assert.fail('Could not land after jumping toward '+x);
}

test('inline scripts parse and photography starts loading before scroll', () => {
  for(const m of html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/g)) {
    if(!m[1].includes('src=')&&!m[1].includes('application/ld+json'))new vm.Script(m[2]);
  }
  assert.match(html, /rel="preload" as="image" href="https:\/\/assets\.unlim8ted\.com\/images\/gallery\/hazy-city\.jpg"/);
  assert.match(html, /loadSceneImage\('photography'\);/);
});

test('single jumps complete the hilly route, recover all signals and reach the story ending', () => {
  const e=engine();e.resetGame();skip(e);
  move(e,380);jump(e,468);move(e,535);jump(e,686);
  move(e,790);jump(e,1040);move(e,1175);jump(e,1260);
  move(e,1650);jump(e,1880);move(e,2010);jump(e,2095);
  move(e,2570);jump(e,2810);move(e,2970);jump(e,3055);move(e,3135);jump(e,3310);move(e,3760);
  assert.ok(e.game.won);assert.equal(e.game.collected.size,6);assert.equal(e.game.scene,'ending');
  assert.deepEqual([...e.game.seen],['opening','ending']);skip(e);assert.equal(e.game.scene,null);
  e.resetGame();assert.equal(e.game.collected.size,0);assert.equal(e.game.scene,'opening');assert.ok(!e.game.won);
});

test('full-speed reversing takes under 100ms and stopping takes under 150ms', () => {
  for(const grounded of [true,false]) {
    const e=engine();e.resetGame();skip(e);Object.assign(e.game,{vx:440,grounded});e.held.add('left');
    for(let i=0;i<12;i++)e.gameStep(1/120);
    assert.ok(e.game.vx<0);assert.ok(e.game.x<155);
  }
  const e=engine();e.resetGame();skip(e);e.game.vx=440;
  for(let i=0;i<18;i++)e.gameStep(1/120);assert.equal(e.game.vx,0);
});

test('release shortens a jump; pressing again in air adds no ability', () => {
  const e=engine();e.resetGame();skip(e);e.held.add('jump');e.game.jumpBuffer=.14;e.gameStep(1/120);
  e.held.clear();e.gameStep(1/120);assert.ok(e.game.vy>=-260);
  const velocity=e.game.vy;e.game.jumpBuffer=.14;e.gameStep(1/120);assert.ok(e.game.vy>velocity);
});

test('slopes retain grounded contact in both directions and arches stay on flat land', () => {
  const e=engine();e.resetGame();skip(e);
  for(const tile of e.ground)for(const direction of [-1,1]) {
    const x=tile.x+tile.w/2;Object.assign(e.game,{x,y:e.groundY(x)-42,vx:direction*440,vy:0,grounded:true});
    e.held.clear();e.held.add(direction<0?'left':'right');e.gameStep(1/120);
    assert.ok(e.game.grounded);assert.ok(Math.abs(e.game.y+42-e.groundY(e.game.x))<.01);
  }
  for(const a of e.archways) {
    const tile=e.ground.find(t=>a.x-a.radius-10>=t.x&&a.x+a.radius+10<=t.x+t.w);
    assert.ok(tile);assert.equal(tile.y,tile.end);assert.equal(tile.y,a.base);
  }
});
