// Offline renderer/stream tests: node --test src/app/tests/test_ask.cjs
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

function renderer() {
  const html = fs.readFileSync(path.join(__dirname, '..', 'quakenet.html'), 'utf8');
  const context = vm.createContext({TextDecoder, TextEncoder, Response, ReadableStream, console});
  vm.runInContext(html.match(/const esc = .*;/)[0] + '\n' + html.match(/const fmtP = .*;/)[0], context);
  vm.runInContext(fs.readFileSync(path.join(__dirname, '..', 'ask.js'), 'utf8'), context);
  return context;
}

test('result cards show tables, escaped values and all additional rows', () => {
  const r = renderer();
  const events = Array.from({length: 12}, (_, i) => ({node: `edge-${i}`, op: '==', value: 1, p: .25, ci95: [.2, .3]}));
  events[0].node = '<img src=x onerror=alert(1)>';
  const html = r.queryResults({events, targets: [{node: 'EdgeOpen::e2', summary: {P: {'0': .25, '1': .75}}}]});
  assert.match(html, /<table/);
  assert.match(html, /95% interval/);
  assert.match(html, /Show 6 more results/);
  assert.match(html, /edge-11/);
  assert.match(html, /EdgeOpen::e2 = 1/);
  assert.ok(!html.includes('<img'));
  assert.match(html, /&lt;img/);
});

test('VOI card shows loss units, no-inspection recommendation and post-inspection actions', () => {
  const r = renderer();
  const html = r.analysisCard({kind: 'voi', loss_units: 'demo units', current_action: 'defer', current_expected_loss: 10,
    recommendation: 'do not inspect', rows: [{node: 'b2', evsi: 0, cost: 2, net_value: -2,
      outcomes: [{reported_state: 0, p: .4, action: 'defer', expected_loss: 10}]}], note: 'Fixture'});
  assert.match(html, /demo units/);
  assert.match(html, /No inspection has positive net value/);
  assert.match(html, /Decisions after each possible outcome/);
  assert.match(html, /-2.00/);
  assert.match(html, /defer/);
});

test('proposal cards keep invalid drafts non-activatable and escape source text', () => {
  const r = renderer();
  const html = r.proposalCard({id: 'draft-1', status: 'draft', spec: {title: 'Test', source: {basis: '<script>bad()</script>'}},
    validation: ['Missing units']}, {activate: true});
  assert.match(html, /Missing units/);
  assert.ok(!html.includes('data-activate'));
  assert.ok(!html.includes('<script>'));
  assert.match(html, /&lt;script&gt;/);
});

test('stream parser handles fragmented UTF-8, final lines without newlines and service errors', async () => {
  const r = renderer();
  const bytes = new TextEncoder().encode(JSON.stringify({type: 'answer', answer: 'n4 → N1'}) + '\n' + JSON.stringify({type: 'plan'}));
  const response = new Response(new ReadableStream({start(controller) {
    for (const byte of bytes) controller.enqueue(new Uint8Array([byte]));
    controller.close();
  }}));
  const events = [];
  await r.consumeAskStream(response, event => events.push(event));
  assert.equal(events[0].answer, 'n4 → N1');
  assert.equal(events[1].type, 'plan');
  await assert.rejects(r.consumeAskStream(new Response('{"error":"Unavailable"}', {status: 503}), () => {}), /Unavailable/);
});

test('conversation export preserves full results, analyses, proposals, review decisions and clarification', () => {
  const r = renderer();
  const turn = {question: 'Which bridge?', assembly: 'toy-ca/demo', model: 'offline fixture', started: '2026-10-22', scenario: {rows: {event: {mw: 7}}},
    queries: [{state: 'computed', statement: 'P(F)', review: 'approved', result: {targets: [{node: 'EdgeOpen::e2', summary: {P: {'0': .25}}}],
      events: Array.from({length: 12}, (_, i) => ({node: `event-${i}`, p: .25}))}}],
    analyses: [{kind: 'compare'}, {kind: 'paths'}], proposals: [{id: 'proposal-1'}], clarification: {question: 'Which definition?'}, answer: 'Choose one.', record: 'ask-1'};
  vm.runInContext(`conversation.push(${JSON.stringify(turn)})`, r);
  const markdown = r.conversationMarkdown();
  for (const text of ['event-11', 'EdgeOpen::e2', 'approved', 'proposal-1', 'Which definition?', 'ask-1', 'toy-ca/demo', '"mw": 7']) assert.ok(markdown.includes(text), text);
});
