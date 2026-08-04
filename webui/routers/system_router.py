from __future__ import annotations

from contention_monitor import ContentionMonitor
from gpu_schedule       import GpuSchedule
from gpu_watchdog       import GpuWatchdog
from notifier           import JobNotifier
from process_manager    import ProcessManager, ProcessNuke, ServerDetacher
from resource_watchdog  import ResourceWatchdog
from routers.dispatch   import HttpExchange, RouteTable, SubRouter
from system_monitor     import SystemMonitor


class SystemRouter(SubRouter):

    def __init__(self, system: SystemMonitor, watchdog: ResourceWatchdog, contention: ContentionMonitor, gpu_guard: GpuWatchdog, gpu_schedule: GpuSchedule, detacher: ServerDetacher, notifier: JobNotifier, nuke: ProcessNuke, processes: ProcessManager) -> None:
        self.system       = system
        self.watchdog     = watchdog
        self.contention   = contention
        self.gpu_guard    = gpu_guard
        self.gpu_schedule = gpu_schedule
        self.detacher     = detacher
        self.notifier     = notifier
        self.nuke         = nuke
        self.processes    = processes

        super().__init__(("/api/system", "/api/gpu-guard", "/api/gpu-schedule", "/api/impact", "/api/notify"))

    def declare(self, table: RouteTable) -> None:
        table.add("GET",  "/api/system",             self.snapshot)
        table.add("GET",  "/api/gpu-guard/history",  self.guard_history)
        table.add("POST", "/api/system/nuke",        self.nuke_everything)
        table.add("POST", "/api/system/detach",      self.detach)
        table.add("POST", "/api/system/shutdown",    self.shutdown)
        table.add("POST", "/api/gpu-schedule",       self.schedule)
        table.add("POST", "/api/impact/arm",         self.arm_impact)
        table.add("POST", "/api/notify/config",      self.notify_config)
        table.add("POST", "/api/notify/test",        self.notify_test)

    def snapshot(self, exchange: HttpExchange) -> None:
        payload                 = self.system.snapshot()
        payload["alerts"]       = self.watchdog.state()
        payload["impact"]       = self.contention.state()
        payload["gpu_guard"]    = self.gpu_guard.state()
        payload["gpu_schedule"] = self.gpu_schedule.state()
        payload["server"]       = self.detacher.state()
        payload["notify"]       = self.notifier.state()

        exchange.send_json(payload)

    def guard_history(self, exchange: HttpExchange) -> None:
        exchange.send_json(self.gpu_guard.history(exchange.integer("limit", "100")))

    def nuke_everything(self, exchange: HttpExchange) -> None:
        self.processes.clear_queue()
        exchange.send_result(self.nuke.nuke())

    def detach(self, exchange: HttpExchange) -> None:
        exchange.send_result(self.detacher.detach(), 500)

    def shutdown(self, exchange: HttpExchange) -> None:
        exchange.send_json({"ok": True, **self.detacher.state()})
        self.detacher.shutdown()

    def schedule(self, exchange: HttpExchange) -> None:
        exchange.send_result(self.gpu_schedule.update(exchange.body))

    def arm_impact(self, exchange: HttpExchange) -> None:
        exchange.send_json(self.contention.arm(bool(exchange.body.get("armed"))))

    def notify_config(self, exchange: HttpExchange) -> None:
        exchange.send_result(self.notifier.configure(exchange.body or {}))

    def notify_test(self, exchange: HttpExchange) -> None:
        exchange.send_result(self.notifier.test())
