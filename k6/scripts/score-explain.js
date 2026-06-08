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
    // name фиксируем шаблоном: иначе k6 берёт URL с уникальным id за имя метрики
    // и плодит по временному ряду на каждый запрос (cardinality explosion).
    const g = http.get(`${BASE}/explain/${id}`, { tags: { ep: 'explain', name: '/explain/{id}' } });
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
  // Сырой дамп k6 data в OUT: и стандартные, и кастомные метрики (score_latency,
  // explain_*) с перцентилями из summaryTrendStats, включая p99 (флаг
  // --summary-export p99 не отдаёт). Разбор и форматирование - в ноутбуке
  // results/analysis.ipynb. Серверные счётчики (queue_dropped, explained)
  // снимает bench.sh из GET /metrics в отдельный <perm>.server.json.
  const out = __ENV.OUT || '/results/summary.json';
  const m = data.metrics;
  const found = m.explain_found ? m.explain_found.values.rate.toFixed(3) : '-';
  const sp99 = m.score_latency ? m.score_latency.values['p(99)'].toFixed(2) : '-';
  const reqs = m.http_reqs ? m.http_reqs.values.count : '?';
  return {
    stdout: `${PERM}: http_reqs=${reqs} found=${found} score_p99=${sp99}ms\n`,
    [out]: JSON.stringify(data),
  };
}
