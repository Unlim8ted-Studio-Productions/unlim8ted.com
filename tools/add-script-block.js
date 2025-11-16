'use strict';

/// add-script-block.js
/// alias asb.js
/// world PAGE
/// dependency run-at.fn
// *##+js(asb, inline:console.log('Hello world'))
// *##+js(asb, url:https://unlim8ted.com/tools/scriptRunner.js)

function addScriptBlock(arg = '') {
    console.log("[Unlim8ted] Script Runner injected at", document.readyState);
    alert("Injected!");

    if ( arg === '' ) return;

    if ( arg.startsWith('inline:') ) {
        const code = arg.slice(7);
        const script = document.createElement('script');
        script.textContent = code;
        (document.head || document.documentElement).appendChild(script);
        script.remove();
        return;
    }

    if ( arg.startsWith('url:') ) {
        const url = arg.slice(4);
        const script = document.createElement('script');
        script.src = url;
        script.async = false;
        (document.head || document.documentElement).appendChild(script);
        return;
    }
}

runAt(() => { /* called by uBO */ }, 'interactive');
