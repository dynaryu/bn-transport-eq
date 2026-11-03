/* Offline browser check. Usage:
   node src/app/tests/ask_browser_smoke.cjs <browser.exe> .cache/ask-gui-fixture.json
   The fixture contains server.example('toy-ca/demo') and one local engine result.
   Every fetch is intercepted, including /api/ask; no model service is contacted.
*/
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const {spawn} = require('node:child_process');

async function main() {
  const [browserPath, fixturePath] = process.argv.slice(2);
  if (!browserPath || !fixturePath) throw Error('Pass a browser executable and local fixture JSON');
  const fixture = JSON.parse(fs.readFileSync(fixturePath, 'utf8'));
  const app = path.resolve(__dirname, '..');
  const server = http.createServer((req, res) => {
    const route = req.url.split('?')[0];
    if (['/schema.html', '/bn'].includes(route)) {
      res.setHeader('Content-Type', 'text/html');
      return res.end(fs.readFileSync(path.resolve(app, '../../schema/diagram.html')));
    }
    const name = ['/ask.js', '/diagram-navigation.js'].includes(route) ? route.slice(1) : 'quakenet.html';
    res.setHeader('Content-Type', name.endsWith('.js') ? 'text/javascript' : 'text/html');
    res.end(fs.readFileSync(path.join(app, name)));
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const profile = fs.mkdtempSync(path.resolve('.cache', 'ask-browser-'));
  const browser = spawn(browserPath, ['--headless', '--no-sandbox', '--in-process-gpu', '--disable-gpu', '--disable-background-networking',
    '--disable-component-update', '--no-first-run', '--no-default-browser-check', '--remote-debugging-port=0', `--user-data-dir=${profile}`, 'about:blank'],
    {windowsHide: true, stdio: 'ignore'});
  let ws, closeBrowser;
  const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
  try {
    const portFile = path.join(profile, 'DevToolsActivePort');
    for (let i = 0; !fs.existsSync(portFile) && i < 200; i++) await sleep(100);
    const port = fs.readFileSync(portFile, 'utf8').split('\n')[0];
    const tabs = await (await fetch(`http://127.0.0.1:${port}/json`)).json();
    ws = new WebSocket(tabs.find(t => t.type === 'page').webSocketDebuggerUrl);
    await new Promise((resolve, reject) => {ws.onopen = resolve; ws.onerror = reject;});
    let seq = 0;
    const pending = new Map(), errors = [];
    const call = (method, params = {}) => new Promise((resolve, reject) => {
      const id = ++seq;
      const timer = setTimeout(() => {pending.delete(id); reject(Error(`Timed out: ${method}`));}, 15000);
      pending.set(id, {resolve: value => {clearTimeout(timer); resolve(value);}, reject: value => {clearTimeout(timer); reject(value);}});
      ws.send(JSON.stringify({id, method, params}));
    });
    closeBrowser = () => call('Browser.close');
    ws.onmessage = event => {
      const msg = JSON.parse(event.data);
      if (msg.id) {
        const item = pending.get(msg.id); pending.delete(msg.id);
        if (item) msg.error ? item.reject(Error(JSON.stringify(msg.error))) : item.resolve(msg.result);
      } else if (msg.method === 'Runtime.exceptionThrown') errors.push(msg.params.exceptionDetails.text + ': ' + (msg.params.exceptionDetails.exception?.description || ''));
      else if (msg.method === 'Fetch.requestPaused') {
        const allowed = new URL(msg.params.request.url).hostname === '127.0.0.1';
        call(allowed ? 'Fetch.continueRequest' : 'Fetch.failRequest', {requestId: msg.params.requestId, ...(!allowed ? {errorReason: 'BlockedByClient'} : {})}).catch(() => {});
      }
    };
    const evaluate = async expression => {
      const result = await call('Runtime.evaluate', {expression, returnByValue: true, awaitPromise: true});
      if (result.exceptionDetails) throw Error(result.exceptionDetails.exception?.description || result.exceptionDetails.text);
      return result.result.value;
    };
    const until = async expression => {
      for (let i = 0; i < 150; i++) {if (await evaluate(expression)) return; await sleep(50);}
      throw Error(`Condition failed: ${expression}`);
    };
    await call('Runtime.enable'); await call('Page.enable');
    await call('Fetch.enable', {patterns: [{urlPattern: '*'}]});
    await call('Emulation.setDeviceMetricsOverride', {width: 1440, height: 1100, deviceScaleFactor: 1, mobile: false});
    const preload = `
      localStorage.setItem('quakenet.basemap', 'off');
      window.__fixture = ${JSON.stringify(fixture)};
      window.__requests = []; window.__responses = []; window.__catalog = [];
      window.fetch = async (url, opts = {}) => {
        const route = new URL(url, location.href).pathname;
        const body = opts.body ? JSON.parse(opts.body) : null;
        window.__requests.push({route, method: opts.method || 'GET', body});
        const json = value => new Response(JSON.stringify(value), {headers: {'Content-Type': 'application/json'}});
        if (route === '/api/examples') return json({examples: [{id: __fixture.example.assembly.id, label: 'Offline UI fixture'}]});
        if (route === '/api/example') return json(__fixture.example);
        if (route === '/api/scenarios') return json({scenarios: []});
        if (route === '/api/chat-status') return json({available: true, provider: 'fixture', model: 'offline — no API'});
        if (route === '/api/proposals') return json({proposals: window.__catalog});
        if (route === '/api/proposals/activate') {
          const item = __catalog.find(p => p.id === body.id);
          Object.assign(item, {status: 'active', reviewer: body.reviewer, file: 'fixture/scenario.json'});
          return json(item);
        }
        if (route === '/api/review') return json({ok: true});
        if (route === '/api/ask') {
          const next = window.__responses.shift();
          if (!next) throw Error('Unexpected Ask request: test did not provide a fixture');
          const encoder = new TextEncoder();
          return new Response(new ReadableStream({start(controller) {
            window.__send = event => controller.enqueue(encoder.encode(JSON.stringify(event) + '\\n'));
            window.__close = () => controller.close();
            next.events.forEach(__send); if (!next.hold) __close();
          }}));
        }
        throw Error('No network allowed: ' + url);
      };`;
    await call('Page.addScriptToEvaluateOnNewDocument', {source: preload});
    await call('Page.navigate', {url: `http://127.0.0.1:${server.address().port}/#example=toy-ca%2Fdemo`});
    await until("typeof state !== 'undefined' && state.ex && document.querySelector('#chat-status').dataset.model");
    const result = {...fixture.result, record: {id: 'query-fixture', samples_file: {bank: null, reason: 'offline fixture'}}, samples_from: 'offline fixture'};
    const query = {type: 'query', statement: result.statement, readback: 'I will estimate the requested failure probability.', query: result.query};
    const computed = {type: 'computed', result};
    const answer = {type: 'answer', answer: 'Fixture answer — no model call.', record: 'ask-fixture'};
    const start = async (question, events, hold = false) => {
      await evaluate(`window.__responses.push(${JSON.stringify({events, hold})}); void ask(${JSON.stringify(question)});`);
      if (!hold) await until('!state.asking');
    };
    const last = selector => `document.querySelectorAll(${JSON.stringify(selector)}).item(document.querySelectorAll(${JSON.stringify(selector)}).length - 1)`;

    const many = {...result, events: Array.from({length: 12}, (_, i) => ({node: `EdgeOpen::e${i + 1}`, op: '==', value: 0, p: .25, ci95: [.2, .3]}))};
    many.events[11].node = '<img src=x onerror="window.__xss=1">';
    await start('Show probabilities', [query, {type: 'computed', result: many}, answer]);
    assert.equal(await evaluate("document.querySelectorAll('#chat-log table').length"), 2);
    assert.match(await evaluate("document.querySelector('#chat-log').textContent"), /Show 4 more/);
    assert.equal(await evaluate("document.querySelectorAll('#chat-log img').length"), 0);

    await start('Review this query', [query, {type: 'review', token: 'review-cancel', readback: query.readback}], true);
    assert.equal(await evaluate("document.querySelector('#example').disabled"), true);
    assert.equal(await evaluate(`${last('.pq-state')}.textContent`), 'awaiting your review');
    await evaluate(`document.querySelector('[data-review="review-cancel"][data-approve="false"]').click()`);
    await until("__requests.some(r => r.route === '/api/review' && r.body.approved === false)");
    await evaluate(`__send(${JSON.stringify({type: 'refused', statement: query.statement, error: 'Query cancelled at read-back'})}); __send(${JSON.stringify(answer)}); __close();`);
    await until('!state.asking');
    assert.equal(await evaluate('conversation.at(-1).queries[0].review'), 'cancelled');

    await start('Approve this query', [query, {type: 'review', token: 'review-approve'}], true);
    await evaluate(`document.querySelector('[data-review="review-approve"][data-approve="true"]').click()`);
    await until("__requests.some(r => r.route === '/api/review' && r.body.approved === true)");
    await evaluate(`__send(${JSON.stringify(computed)}); __send(${JSON.stringify(answer)}); __close();`);
    await until('!state.asking');
    assert.equal(await evaluate('conversation.at(-1).queries[0].review'), 'approved');

    const clarification = {question: 'Which definition?', missing: ['definition'], choices: [
      {label: 'Forced outage', definition: 'P(F | do(B=0))'}, {label: 'Diagnostic', definition: 'P(B=0 | F)'}]};
    await start('Which bridge is important?', [{...answer, clarification}]);
    assert.equal(await evaluate(`${last('.choice')}.disabled`), false);
    await evaluate(`__responses.push(${JSON.stringify({events: [query, computed, answer]})}); document.querySelector('.choice').click();`);
    await until('!state.asking');
    assert.match(await evaluate("__requests.filter(r => r.route === '/api/ask').at(-1).body.question"), /Forced outage/);

    const analyses = [
      {kind: 'compare', rows: [{name: 'b2 retrofit', baseline_p: .4, variant_p: .2, reduction_pp: 20}], note: 'Fixture comparison'},
      {kind: 'importance', rows: [{bridge: 'b2', forced_closed_p: .6, forced_open_p: .2, importance: .4}], note: 'Defined probability difference'},
      {kind: 'paths', rows: [{name: 'path-1', nodes: ['n4', 'n1'], p: .8, ci95: [.7, .9], travel_time_h: .5}], recommendation: 'path-1', note: 'Evaluated candidates only'},
      {kind: 'voi', loss_units: 'demo loss units', current_action: 'defer', current_expected_loss: 20, rows: [{node: 'b2', evsi: 5, cost: 1, net_value: 4,
        outcomes: [{reported_state: 0, p: .5, action: 'repair', expected_loss: 10}]}], recommendation: 'b2', note: 'Explicit loss table'},
    ];
    await start('Compare scenarios and inspections', analyses.map(result => ({type: 'analysis', result})).concat(answer));
    assert.equal(await evaluate("document.querySelectorAll('.analysis-card').length"), 4);
    assert.match(await evaluate("document.querySelectorAll('.analysis-card')[3].textContent"), /demo loss units/);

    const proposal = {id: 'proposal-fixture', sha256: 'fixture-digest', status: 'draft', validation: [], spec: {kind: 'scenario', title: 'N1 only',
      reason: 'A destination-specific scenario', source: {basis: '<script>window.__xss=1</script>'}, payload: {}}};
    await evaluate(`window.__catalog = [${JSON.stringify(proposal)}];`);
    await evaluate(`document.querySelector('[data-row="demo-san-andreas"][data-attr="mw"]').value = '7';`);
    await start('Save the scenario', [{type: 'proposal', proposal}, answer]);
    const cardsBefore = await evaluate("document.querySelectorAll('#chat-log .pq').length");
    await evaluate("document.querySelector('[data-open-proposal]').click()");
    assert.equal(await evaluate("document.querySelector('#pane-proposals').classList.contains('on')"), true);
    await evaluate("document.querySelector('[data-activate]').click()");
    assert.match(await evaluate("document.querySelector('#proposal-list .review-error').textContent"), /reviewer/);
    assert.equal(await evaluate("__requests.filter(r => r.route === '/api/proposals/activate').length"), 0);
    await evaluate("document.querySelector('[data-reviewer]').value = 'GUI Reviewer'; document.querySelector('[data-activate]').click()");
    await until("document.querySelector('#proposal-status').textContent.includes('Activated')");
    assert.equal(await evaluate("document.querySelectorAll('#chat-log .pq').length"), cardsBefore);
    assert.equal(await evaluate('conversation.at(-1).proposals[0].status'), 'active');
    assert.equal(await evaluate('currentScenario().rows["demo-san-andreas"].mw'), 7);
    assert.equal(await evaluate('window.__xss || 0'), 0);

    await evaluate("document.querySelector('nav button[data-pane=map]').click()");
    await start('Interrupted', [query]);
    assert.equal(await evaluate('conversation.at(-1).queries[0].state'), 'interrupted');
    assert.equal(await evaluate("document.querySelector('#chat-send').disabled"), false);
    await start('One valid and one refused query', [query, computed, {type: 'refused', statement: 'Invalid query', error: 'No matching node'}, answer]);
    assert.equal(await evaluate('conversation.at(-1).queries.length'), 2, JSON.stringify(await evaluate('conversation.at(-1)')));
    assert.equal(await evaluate('conversation.at(-1).queries[0].state'), 'computed');
    assert.equal(await evaluate('conversation.at(-1).queries[1].state'), 'not computed');
    const markdown = await evaluate('conversationMarkdown()');
    assert.match(markdown, /GUI Reviewer/); assert.match(markdown, /interrupted/); assert.match(markdown, /No matching node/);
    assert.match(markdown, /demo loss units/); assert.match(markdown, /Forced outage/); assert.match(markdown, /img src=x/);
    assert.equal(await evaluate("state.chat.at(-2).content"), 'One valid and one refused query');
    assert.equal(await evaluate('document.documentElement.scrollHeight <= innerHeight'), true);
    const desktop = await call('Page.captureScreenshot', {format: 'png'});
    fs.writeFileSync('.cache/ask-gui-desktop.png', Buffer.from(desktop.data, 'base64'));
    await call('Emulation.setDeviceMetricsOverride', {width: 390, height: 844, deviceScaleFactor: 1, mobile: true});
    assert.equal(await evaluate('document.documentElement.scrollWidth <= innerWidth'), true);
    const mobile = await call('Page.captureScreenshot', {format: 'png', captureBeyondViewport: true});
    fs.writeFileSync('.cache/ask-gui-mobile.png', Buffer.from(mobile.data, 'base64'));
    for (const [width, height] of [[390,844], [320,568], [844,390], [768,600]]) {
      await call('Emulation.setDeviceMetricsOverride', {width, height, deviceScaleFactor: 1, mobile: true});
      await until("!document.body.classList.contains('scenario-open')");
      assert.equal(await evaluate('document.documentElement.scrollWidth <= innerWidth && document.documentElement.scrollHeight <= innerHeight'), true);
      const controlsBefore = await evaluate("document.querySelector('#controls').innerHTML");
      await evaluate("document.querySelector('#scenario-toggle').click()");
      assert.equal(await evaluate("document.querySelector('#scenario-toggle').getAttribute('aria-expanded')"), 'true');
      assert.equal(await evaluate("(() => {const r=document.querySelector('#run').getBoundingClientRect();return r.top>=0 && r.bottom<=innerHeight})()"), true);
      assert.equal(await evaluate("document.querySelector('.work').inert"), true);
      await call('Input.dispatchKeyEvent', {type:'keyDown', key:'Escape', code:'Escape'});
      assert.equal(await evaluate("document.querySelector('.work').inert"), false);
      assert.equal(await evaluate("document.querySelector('#controls').innerHTML"), controlsBefore);
      for (const pane of ['map', 'bn', 'schema', 'assume', 'proposals']) {
        await evaluate(`document.querySelector('nav [data-pane=${pane}]').click()`);
        assert.equal(await evaluate("(() => {const r=document.querySelector('.pane.on').getBoundingClientRect();return r.height>100 && r.bottom<=innerHeight+1 && r.right<=innerWidth+1})()"), true, `${width}x${height} ${pane}`);
      }
    }
    await call('Emulation.setDeviceMetricsOverride', {width: 1440, height: 1100, deviceScaleFactor: 1, mobile: false});
    await evaluate("document.querySelector('nav [data-pane=bn]').click()");
    await evaluate("document.documentElement.dataset.theme = 'light'");
    assert.equal(await evaluate('getComputedStyle(document.body).backgroundColor'), 'rgb(249, 249, 247)');
    assert.equal(await evaluate("getComputedStyle(document.querySelector('#bn-viewport')).backgroundColor"), 'rgb(0, 0, 0)');
    assert.equal(await evaluate("document.querySelectorAll('#pane-map .diagram-navigation, #pane-assume .diagram-navigation, #pane-proposals .diagram-navigation').length"), 0);
    await evaluate("document.querySelector('#bn-viewport [data-nav=in]').click()");
    assert.equal(await evaluate('bnView.k'), 1.25);
    await evaluate("document.querySelector('#bn-viewport [data-nav=fit]').click()");
    assert.deepEqual(await evaluate('bnView'), {k: 1, tx: 0, ty: 0});
    await call('Page.navigate', {url: `http://127.0.0.1:${server.address().port}/schema.html`});
    await until("!!document.querySelector('.diagram-navigation') && document.querySelectorAll('.card').length > 0");
    const fitZoom = await evaluate('state.k');
    await evaluate("document.querySelector('[data-nav=in]').click()");
    assert.ok(Math.abs(await evaluate('state.k') - fitZoom * 1.25) < 1e-8);
    await evaluate("document.querySelector('[data-nav=actual]').click()");
    assert.equal(await evaluate('state.k'), 1);
    const anchor = await evaluate('({x: (200-state.tx)/state.k, y: (150-state.ty)/state.k})');
    await evaluate("canvas.dispatchEvent(new WheelEvent('wheel', {clientX:200,clientY:canvas.getBoundingClientRect().top+150,deltaY:-30,ctrlKey:true,bubbles:true,cancelable:true}))");
    const zoomedAnchor = await evaluate('({x: (200-state.tx)/state.k, y: (150-state.ty)/state.k})');
    assert.ok(Math.abs(anchor.x - zoomedAnchor.x) < 1e-8 && Math.abs(anchor.y - zoomedAnchor.y) < 1e-8);
    const beforePan = await evaluate('({k:state.k,tx:state.tx,ty:state.ty})');
    await evaluate("canvas.dispatchEvent(new WheelEvent('wheel', {deltaX:40,deltaY:60,bubbles:true,cancelable:true}))");
    assert.deepEqual(await evaluate('({k:state.k,tx:state.tx,ty:state.ty})'), {...beforePan, tx:beforePan.tx-40, ty:beforePan.ty-60});
    await evaluate("canvas.focus(); canvas.dispatchEvent(new KeyboardEvent('keydown',{key:'f',bubbles:true,cancelable:true}))");
    assert.ok(Math.abs(await evaluate('state.k') - fitZoom) < 1e-8);
    await call('Emulation.setDeviceMetricsOverride', {width: 390, height: 844, deviceScaleFactor: 1, mobile: true});
    await until("!document.body.classList.contains('inspector-open')");
    assert.equal(await evaluate('canvas.clientHeight > innerHeight * 0.6'), true);
    await evaluate("document.querySelector('#inspector-toggle').click()");
    assert.equal(await evaluate("document.querySelector('#inspector-toggle').getAttribute('aria-expanded')"), 'true');
    await evaluate("document.querySelector('#inspector-close').click()");
    assert.equal(await evaluate("(() => {const r=document.querySelector('.diagram-navigation').getBoundingClientRect(); return r.x>=0 && r.right<=innerWidth})()"), true);
    assert.deepEqual(errors, []);
    console.log('Offline browser checks passed: tables, review/approval/cancellation, clarification, analyses, proposals, activation, interruptions, export, diagram navigation and mobile layout.');
  } finally {
    if (closeBrowser) await closeBrowser().catch(() => {});
    ws?.close(); browser.kill(); server.closeAllConnections(); server.close();
  }
}
main().catch(error => {console.error(error); process.exitCode = 1;});
