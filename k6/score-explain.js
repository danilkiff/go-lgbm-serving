// Смешанная нагрузка decline->explain через REST. Каждая итерация шлёт POST
// /score случайным вектором; на decision=decline опрашивает GET /explain/{id} до
// готовности объяснения (оно считается асинхронно). Так бьются все три участка,
// где бэкенд хранилища вообще проявляется: запись объяснения воркером, его
// чтение через GET и способность воркеров поспевать за потоком отклонений.
// Горячий путь /score хранилище не трогает - его латентность здесь как контроль.
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Trend, Rate, Counter } from 'k6/metrics';
import { SharedArray } from 'k6/data';

// GET /explain/{id} легитимно отдаёт 404, пока объяснение ещё не посчитано
// (согласованность в конечном счёте). Объявляем 404 ожидаемым, иначе эти опросы
// раздувают http_req_failed и маскируют настоящие сбои.
http.setResponseCallback(http.expectedStatuses(200, 404));

const rows = new SharedArray('feature rows', () => JSON.parse(open('/scripts/rows.json')));

const BASE = __ENV.BASE_URL || 'http://localhost:8080';
const PERM = __ENV.PERM || 'perm';
const TRIES = Number(__ENV.EXPLAIN_TRIES || 15);
const BACKOFF = Number(__ENV.EXPLAIN_BACKOFF || 0.005); // сек между опросами GET

const scoreLatency = new Trend('score_latency', true);   // горячий путь, контроль
const explainGet = new Trend('explain_latency', true);   // RTT успешного GET = чтение из хранилища
const explainWait = new Trend('explain_wait', true);     // от 1-го опроса до готовности (очередь+SHAP+запись)
const explainFound = new Rate('explain_found');          // доля отклонений, объяснённых в срок
const explainMiss = new Counter('explain_poll_misses');

export const options = {
  scenarios: {
    mixed: {
      executor: 'constant-arrival-rate',
      rate: Number(__ENV.RATE || 400),
      timeUnit: '1s',
      duration: __ENV.DURATION || '60s',
      preAllocatedVUs: Number(__ENV.VUS || 100),
      maxVUs: Number(__ENV.MAX_VUS || 600),
    },
  },
  thresholds: {
    http_req_failed: ['rate<0.05'], // информативно, прогон не прерываем
  },
  summaryTrendStats: ['avg', 'med', 'p(95)', 'p(99)', 'max'],
};

export default function () {
  const row = rows[Math.floor(Math.random() * rows.length)];
  const res = http.post(`${BASE}/score`, JSON.stringify({ features: row }), {
    headers: { 'Content-Type': 'application/json' },
    tags: { ep: 'score' },
  });
  scoreLatency.add(res.timings.duration);
  if (!check(res, { 'score 200': (r) => r.status === 200 })) return;

  let body;
  try { body = res.json(); } catch (e) { return; }
  if (body.decision !== 'decline') return;

  const id = body.id;
  let found = false;
  let waited = 0;
  for (let i = 0; i < TRIES; i++) {
    const g = http.get(`${BASE}/explain/${id}`, { tags: { ep: 'explain' } });
    if (g.status === 200) {
      explainGet.add(g.timings.duration);
      explainWait.add(waited * 1000 + g.timings.duration);
      found = true;
      break;
    }
    explainMiss.add(1);
    sleep(BACKOFF);
    waited += BACKOFF;
  }
  explainFound.add(found);
}

export function handleSummary(data) {
  const m = data.metrics;
  const v = (name, key) => (m[name] && m[name].values[key] !== undefined ? m[name].values[key] : null);
  const r2 = (x) => (x === null ? null : Math.round(x * 100) / 100);
  const trend = (name) => ({ avg: r2(v(name, 'avg')), p95: r2(v(name, 'p(95)')), p99: r2(v(name, 'p(99)')), max: r2(v(name, 'max')) });

  const summary = {
    perm: PERM,
    store: __ENV.STORE || null,
    config: { rate: Number(__ENV.RATE || 400), duration: __ENV.DURATION || '60s' },
    score_ms: trend('score_latency'),
    explain_get_ms: trend('explain_latency'),
    explain_wait_ms: trend('explain_wait'),
    explain_found_rate: r2(v('explain_found', 'rate')),
    explain_poll_misses: v('explain_poll_misses', 'count'),
    http_reqs: v('http_reqs', 'count'),
    http_req_rate: r2(v('http_reqs', 'rate')),
    http_req_failed_rate: v('http_req_failed', 'rate'),
    iterations: v('iterations', 'count'),
    vus_max: v('vus_max', 'value'),
  };

  const text =
    `\n=== ${PERM} (store=${summary.store}) ===\n` +
    `score    ms  ${JSON.stringify(summary.score_ms)}\n` +
    `explainGET   ${JSON.stringify(summary.explain_get_ms)}\n` +
    `explainWAIT  ${JSON.stringify(summary.explain_wait_ms)}\n` +
    `explain_found=${summary.explain_found_rate}  misses=${summary.explain_poll_misses}\n` +
    `http_reqs=${summary.http_reqs} (${summary.http_req_rate}/s)  failed=${summary.http_req_failed_rate}  iters=${summary.iterations}  vus_max=${summary.vus_max}\n`;

  const out = { stdout: text };
  out[`/results/${PERM}.json`] = JSON.stringify(summary, null, 2);
  return out;
}
