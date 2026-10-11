import assert from "node:assert";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

class FakeEventSource {
  static CONNECTING = 0;
  static OPEN       = 1;
  static CLOSED     = 2;
  static opened     = [];

  constructor(url) {
    this.url        = url;
    this.readyState = FakeEventSource.CONNECTING;
    this.closed     = false;
    this.onmessage  = null;
    this.onerror    = null;
    FakeEventSource.opened.push(this);
  }

  hello(boot) {
    this.readyState = FakeEventSource.OPEN;
    this.message({ type: "hello", channel: "x", boot });
  }

  message(data) {
    this.onmessage({ data: JSON.stringify(data) });
  }

  fail(terminal) {
    this.readyState = terminal ? FakeEventSource.CLOSED : FakeEventSource.CONNECTING;
    this.onerror();
  }

  close() {
    this.closed     = true;
    this.readyState = FakeEventSource.CLOSED;
  }
}


class StreamHubClientSuite {
  constructor(source) {
    this.posts  = [];
    this.answer = () => ({ ok: true });

    const sandbox = {
      window      : {},
      crypto      : { getRandomValues: (array) => array.fill(7) },
      EventSource : FakeEventSource,
      Api         : { post: async (url, body) => { this.posts.push({ url, body }); return this.answer(url, body); } },
      setTimeout, clearTimeout, JSON, Map, Set, Array, Uint8Array, Promise, console,
    };
    vm.createContext(sandbox);
    vm.runInContext(source, sandbox, { filename: "stream_hub.js" });

    this.StreamHub = sandbox.window.StreamHub;
  }

  tick() {
    return new Promise((resolve) => setImmediate(resolve));
  }

  fresh() {
    FakeEventSource.opened.length = 0;
    this.posts.length = 0;
    this.answer = () => ({ ok: true });
    return new this.StreamHub();
  }

  current() {
    return FakeEventSource.opened[FakeEventSource.opened.length - 1];
  }

  async testTheFirstSubscriptionOpensTheSingleChannelAndAnnouncesAfterHello() {
    const hub = this.fresh();
    const sub = hub.subscribe("job-1", null, () => {});

    assert.equal(FakeEventSource.opened.length, 1);
    assert.equal(this.current().url, `/api/stream?channel=${hub.channel}`);
    assert.equal(this.posts.length, 0, "nothing is announced before the hello frame");

    this.current().hello("boot-a");
    await this.tick();
    assert.deepEqual(this.posts, [{ url: `/api/stream/${hub.channel}/subscribe`, body: { job_id: "job-1", unit: null, after: 0 } }]);
    assert.equal(hub.connected, true);
    assert.equal(sub.key, "job-1");
  }

  async testEventsAreRoutedByKeyAndTheLastIdIsRememberedPerSubscription() {
    const hub = this.fresh();
    const a   = [];
    const b   = [];
    hub.subscribe("job-a", null, (e) => a.push(e));
    hub.subscribe("job-b", 2,    (e) => b.push(e));
    this.current().hello("boot-a");
    await this.tick();

    this.current().message({ key: "job-a", type: "chunk", data: "A", id: 5 });
    this.current().message({ key: "job-b/2", type: "chunk", data: "B", id: 1 });
    this.current().message({ key: "job-zzz", type: "chunk", data: "nobody", id: 9 });

    assert.deepEqual(a.map((e) => e.data), ["A"]);
    assert.deepEqual(b.map((e) => e.data), ["B"]);
    assert.equal(hub.keys.get("job-a").lastId, 5);
    assert.equal(hub.keys.get("job-b/2").lastId, 1);
  }

  async testASecondSubscriptionOnAnOpenChannelIsAnnouncedAtOnce() {
    const hub = this.fresh();
    hub.subscribe("job-a", null, () => {});
    this.current().hello("boot-a");
    await this.tick();
    this.posts.length = 0;

    hub.subscribe("job-a", 3, () => {});
    await this.tick();

    assert.equal(FakeEventSource.opened.length, 1, "no second connection");
    assert.deepEqual(this.posts.map((p) => p.body), [{ job_id: "job-a", unit: 3, after: 0 }]);
  }

  async testSevenViewersShareOneConnection() {
    const hub = this.fresh();
    for (let index = 0; index < 7; index++) hub.subscribe(`job-${index}`, null, () => {});
    this.current().hello("boot-a");
    await this.tick();

    assert.equal(FakeEventSource.opened.length, 1, "seven tiles hold a single EventSource");
    assert.equal(this.posts.filter((p) => p.url.endsWith("/subscribe")).length, 7);
  }

  async testClosingTheLastSubscriptionPostsUnsubscribeAndDropsTheConnection() {
    const hub = this.fresh();
    const sub = hub.subscribe("job-a", null, () => {});
    this.current().hello("boot-a");
    await this.tick();
    this.posts.length = 0;

    sub.close();
    await this.tick();

    assert.deepEqual(this.posts, [{ url: `/api/stream/${hub.channel}/unsubscribe`, body: { job_id: "job-a", unit: null } }]);
    assert.equal(this.current().closed, true);
    assert.equal(hub.source, null);
    assert.equal(hub.connected, false);
    assert.equal(sub.closed, true);

    sub.close();
    assert.equal(this.posts.length, 1, "closing twice posts once");
  }

  async testADroppedConnectionNotifiesEverySubscriptionOnceAndReannouncesFromTheLastIdAfterHello() {
    const hub  = this.fresh();
    const seen = [];
    hub.subscribe("job-a", null, (e) => seen.push(e));
    this.current().hello("boot-a");
    await this.tick();
    this.current().message({ key: "job-a", type: "chunk", data: "x", id: 42 });
    this.posts.length = 0;

    this.current().fail(false);
    this.current().fail(false);
    assert.deepEqual(seen.filter((e) => e.type === "link"), [{ type: "link", state: "lost" }]);
    assert.equal(hub.connected, false);
    assert.equal(FakeEventSource.opened.length, 1, "the browser's own retry is left to the EventSource");

    this.current().hello("boot-a");
    await this.tick();
    assert.deepEqual(this.posts.map((p) => p.body), [{ job_id: "job-a", unit: null, after: 42 }]);
  }

  async testARestartedServerResetsEveryLastIdSoTheReplayStartsOver() {
    const hub = this.fresh();
    hub.subscribe("job-a", null, () => {});
    this.current().hello("boot-a");
    await this.tick();
    this.current().message({ key: "job-a", type: "chunk", data: "x", id: 42 });
    this.posts.length = 0;

    this.current().fail(false);
    this.current().hello("boot-b");
    await this.tick();

    assert.deepEqual(this.posts.map((p) => p.body), [{ job_id: "job-a", unit: null, after: 0 }]);
  }

  async testATerminallyClosedConnectionIsReopenedAfterTheRetryDelayWhileSubscriptionsRemain() {
    this.StreamHub.RETRY_MS = 5;
    const hub = this.fresh();
    hub.subscribe("job-a", null, () => {});
    this.current().hello("boot-a");
    await this.tick();

    this.current().fail(true);
    assert.equal(hub.source, null);
    await new Promise((resolve) => setTimeout(resolve, 20));

    assert.equal(FakeEventSource.opened.length, 2, "a new EventSource was opened");
    assert.equal(FakeEventSource.opened[0].closed, true);
    this.StreamHub.RETRY_MS = 3000;
  }

  async testARefusedSubscriptionReachesItsHandlerAsUnavailableAStaleChannelDoesNot() {
    const hub  = this.fresh();
    const seen = [];
    this.answer = (url, body) => (body.job_id === "ghost" ? { ok: false, error: "unknown job" } : { ok: false, error: "unknown channel", stale: true });
    hub.subscribe("ghost", null, (e) => seen.push(e));
    hub.subscribe("later", null, (e) => seen.push(e));
    this.current().hello("boot-a");
    await this.tick();
    await this.tick();

    assert.deepEqual(seen, [{ type: "unavailable", error: "unknown job" }]);
  }

  async testATruncationMarkerFansOutToEverySubscription() {
    const hub = this.fresh();
    const a   = [];
    const b   = [];
    hub.subscribe("job-a", null, (e) => a.push(e));
    hub.subscribe("job-b", null, (e) => b.push(e));
    this.current().hello("boot-a");
    await this.tick();

    this.current().message({ key: null, type: "truncated", dropped: 12 });

    assert.deepEqual(a, [{ key: null, type: "truncated", dropped: 12 }]);
    assert.deepEqual(b, [{ key: null, type: "truncated", dropped: 12 }]);
  }

  async testTwoViewersOfOneKeyShareASingleServerSubscriptionAndBothReceiveItsEvents() {
    const hub   = this.fresh();
    const tile  = [];
    const tuner = [];
    const a     = hub.subscribe("job-a", null, (e) => tile.push(e));
    this.current().hello("boot-a");
    await this.tick();
    const b = hub.subscribe("job-a", null, (e) => tuner.push(e));
    await this.tick();

    assert.equal(FakeEventSource.opened.length, 1, "no second connection");
    this.current().message({ key: "job-a", type: "chunk", data: "x", id: 1 });
    assert.deepEqual(tile.map((e) => e.data), ["x"]);
    assert.deepEqual(tuner.map((e) => e.data), ["x"]);
    assert.equal(a.closed, false);

    this.posts.length = 0;
    a.close();
    assert.equal(this.posts.length, 0, "the key stays subscribed while a viewer remains");
    this.current().message({ key: "job-a", type: "chunk", data: "y", id: 2 });
    assert.deepEqual(tile.map((e) => e.data), ["x"], "a closed viewer hears nothing more");
    assert.deepEqual(tuner.map((e) => e.data), ["x", "y"]);

    b.close();
    await this.tick();
    assert.deepEqual(this.posts.map((p) => p.url.split("/").pop()), ["unsubscribe"]);
    assert.equal(hub.source, null, "the channel closes with the last viewer");
  }

  async testAViewerJoiningAFollowedKeyIsReplayedTheWholeStreamWhileTheFirstSeesNothingTwice() {
    const hub   = this.fresh();
    const tuner = [];
    const tile  = [];
    hub.subscribe("job-a", null, (e) => tuner.push(e.id));
    this.current().hello("boot-a");
    await this.tick();
    for (const id of [1, 2, 3]) this.current().message({ key: "job-a", type: "chunk", data: "x", id });
    this.posts.length = 0;

    hub.subscribe("job-a", null, (e) => tile.push(e.id));
    await this.tick();
    assert.deepEqual(this.posts.map((p) => p.body), [{ job_id: "job-a", unit: null, after: 0 }], "the late viewer re-announces the key from the start");

    this.current().message({ key: "job-a", type: "chunk", data: "in flight", id: 4 });
    for (const id of [1, 2, 3, 4, 5]) this.current().message({ key: "job-a", type: "chunk", data: "replay", id });

    assert.deepEqual(tuner, [1, 2, 3, 4, 5], "the first viewer drops the replayed ids it already holds");
    assert.deepEqual(tile,  [1, 2, 3, 4, 5], "the late viewer waits for the replay instead of starting mid-stream");
  }

  async testAViewerJoiningWhileTheLinkIsDownIsReplayedFromTheStartAfterHello() {
    const hub  = this.fresh();
    const tile = [];
    hub.subscribe("job-a", null, () => {});
    this.current().hello("boot-a");
    await this.tick();
    for (const id of [1, 2, 3]) this.current().message({ key: "job-a", type: "chunk", data: "x", id });
    this.current().fail(false);
    this.posts.length = 0;

    hub.subscribe("job-a", null, (e) => tile.push(e.id));
    await this.tick();
    assert.equal(this.posts.length, 0, "nothing is announced while the link is down");

    this.current().hello("boot-a");
    await this.tick();
    assert.deepEqual(this.posts.map((p) => p.body), [{ job_id: "job-a", unit: null, after: 0 }]);

    for (const id of [1, 2, 3, 4]) this.current().message({ key: "job-a", type: "chunk", data: "replay", id });
    assert.deepEqual(tile, [1, 2, 3, 4]);
  }

  async testARestartClearsTheGateOfAViewerStillWaitingForItsReplay() {
    const hub  = this.fresh();
    const tile = [];
    hub.subscribe("job-a", null, () => {});
    this.current().hello("boot-a");
    await this.tick();
    for (const id of [1, 2, 3, 4, 5, 6]) this.current().message({ key: "job-a", type: "chunk", data: "x", id });
    hub.subscribe("job-a", null, (e) => { if (e.id != null) tile.push(e.id); });
    await this.tick();

    this.current().fail(false);
    this.current().hello("boot-b");
    await this.tick();
    for (const id of [1, 2]) this.current().message({ key: "job-a", type: "chunk", data: "after restart", id });

    assert.deepEqual(tile, [1, 2]);
  }

  async testUnitTailsAreNotDeduplicatedSinceEveryAnnounceRestartsTheirIds() {
    const hub  = this.fresh();
    const seen = [];
    hub.subscribe("job-a", 2, (e) => { if (e.id != null) seen.push(e.id); });
    this.current().hello("boot-a");
    await this.tick();
    for (const id of [1, 2, 3]) this.current().message({ key: "job-a/2", type: "chunk", data: "x", id });

    this.current().fail(false);
    this.current().hello("boot-a");
    await this.tick();
    for (const id of [1, 2]) this.current().message({ key: "job-a/2", type: "chunk", data: "fresh tail", id });

    assert.deepEqual(seen, [1, 2, 3, 1, 2]);
  }

  async testAViewerClosingItselfInsideItsHandlerDoesNotDisturbTheOtherViewersOfTheKey() {
    const hub  = this.fresh();
    const seen = [];
    const self = hub.subscribe("job-a", null, (e) => { seen.push("self:" + e.type); if (e.type === "end") self.close(); });
    hub.subscribe("job-a", null, (e) => seen.push("other:" + e.type));
    this.current().hello("boot-a");
    await this.tick();

    this.current().message({ key: "job-a", type: "end", id: 3 });

    assert.deepEqual(seen, ["self:end", "other:end"]);
    assert.equal(hub.keys.get("job-a").subs.size, 1);
    assert.equal(hub.connected, true);
  }

  async testMalformedFramesAreIgnored() {
    const hub  = this.fresh();
    const seen = [];
    hub.subscribe("job-a", null, (e) => seen.push(e));
    this.current().hello("boot-a");
    await this.tick();

    this.current().onmessage({ data: "{not json" });
    assert.deepEqual(seen, []);
  }

  async run() {
    const cases = Object.getOwnPropertyNames(StreamHubClientSuite.prototype).filter((name) => name.startsWith("test"));
    let failed  = 0;

    for (const name of cases) {
      try {
        await this[name]();
        process.stdout.write(`ok   ${name}\n`);
      } catch (error) {
        failed += 1;
        process.stdout.write(`FAIL ${name}\n     ${error.message}\n`);
      }
    }

    process.stdout.write(failed ? `${failed} failing\n` : `all ${cases.length} passing\n`);
    process.exit(failed ? 1 : 0);
  }
}

const here = path.dirname(fileURLToPath(import.meta.url));
await new StreamHubClientSuite(fs.readFileSync(path.join(here, "..", "..", "webui", "static", "js", "stream_hub.js"), "utf-8")).run();
