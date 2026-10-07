from __future__ import annotations

from cube_explorer    import CubeExplorer, SliceCollectionMailer, SliceCollector
from routers.dispatch import HttpExchange, RouteTable, SubRouter


class CubeRouter(SubRouter):

    def __init__(self, cubes: CubeExplorer) -> None:
        self.cubes = cubes

        super().__init__(("/api/cubes",))

    def declare(self, table: RouteTable) -> None:
        table.add("GET", "/api/cubes",              self.listing)
        table.add("GET", "/api/cubes/status",       self.status)
        table.add("GET", "/api/cubes/primary",      self.primary)
        table.add("GET", "/api/cubes/profiles",     self.profiles)
        table.add("GET", "/api/cubes/ssim",         self.ssim)
        table.add("GET", "/api/cubes/plane",        self.plane)
        table.add("GET", "/api/cubes/cbar",         self.cbar)
        table.add("GET", "/api/cubes/metric_map",   self.metric_map)
        table.add("GET", "/api/cubes/metric_at",    self.metric_at)
        table.add("GET", "/api/cubes/selective",    self.selective)
        table.add("GET", "/api/cubes/points",       self.points)
        table.add("GET", "/api/cubes/aux_grid",     self.aux_grid)
        table.add("GET", "/api/cubes/transect",     self.transect)
        table.add("GET", "/api/cubes/param_map",    self.param_map)
        table.add("GET", "/api/cubes/param_cbar",   self.param_cbar)
        table.add("GET", "/api/cubes/params_at",    self.params_at)
        table.add("GET", "/api/cubes/slice",        self.slice_view)

        table.add("POST", "/api/cubes/load",           self.load)
        table.add("POST", "/api/cubes/attach",         self.attach)
        table.add("POST", "/api/cubes/detach",         self.detach)
        table.add("POST", "/api/cubes/save_transect",  self.save_transect)
        table.add("POST", "/api/cubes/save_slices",    self.save_slices)

    def listing(self, exchange: HttpExchange) -> None:
        exchange.send_json(self.cubes.list_cubes(exchange.text("base")))

    def status(self, exchange: HttpExchange) -> None:
        exchange.send_json(self.cubes.load_status())

    def primary(self, exchange: HttpExchange) -> None:
        exchange.send_png(self.cubes.primary_png(exchange.text("id")))

    def profiles(self, exchange: HttpExchange) -> None:
        result = self.cubes.profiles(
            cube_id = exchange.text("id"),
            row     = exchange.integer("row"),
            col     = exchange.integer("col"),
        )
        exchange.send_result(result, 404)

    def ssim(self, exchange: HttpExchange) -> None:
        result = self.cubes.slice_ssim(
            cube_id = exchange.text("id"),
            row     = exchange.integer("row"),
            col     = exchange.integer("col"),
            space   = exchange.text("space", "physical"),
        )
        exchange.send_result(result, 404)

    def plane(self, exchange: HttpExchange) -> None:
        png = self.cubes.plane_png(
            cube_id = exchange.text("id"),
            source  = exchange.text("source", "pred"),
            frac    = exchange.number("frac"),
            space   = exchange.text("space", "physical"),
            cmap    = exchange.text("cmap", "jet"),
        )
        exchange.send_png(png)

    def cbar(self, exchange: HttpExchange) -> None:
        exchange.send_png(self.cubes.cbar_png(exchange.text("cmap", "viridis")))

    def metric_map(self, exchange: HttpExchange) -> None:
        png = self.cubes.metric_overlay_png(
            cube_id  = exchange.text("id"),
            key      = exchange.text("key"),
            vmin     = exchange.number("vmin"),
            vmax     = exchange.number("vmax"),
            keep_min = exchange.number("keep_min", "-inf"),
            keep_max = exchange.number("keep_max", "inf"),
            alpha    = exchange.number("alpha", "0.75"),
        )
        exchange.send_png(png)

    def metric_at(self, exchange: HttpExchange) -> None:
        result = self.cubes.metric_value_at(
            cube_id = exchange.text("id"),
            key     = exchange.text("key"),
            row     = exchange.integer("row"),
            col     = exchange.integer("col"),
        )
        exchange.send_result(result, 404)

    def selective(self, exchange: HttpExchange) -> None:
        result = self.cubes.selective_metrics(
            cube_id  = exchange.text("id"),
            key      = exchange.text("key"),
            coverage = exchange.number("coverage", "1.0"),
        )
        exchange.send_result(result)

    def points(self, exchange: HttpExchange) -> None:
        blob = self.cubes.points_bin(
            cube_id    = exchange.text("id"),
            source     = exchange.text("source", "pred"),
            amp_min    = exchange.number("amp_min", "0.001"),
            max_points = exchange.integer("max", "60000"),
        )
        exchange.send_bytes(blob)

    def aux_grid(self, exchange: HttpExchange) -> None:
        exchange.send_bytes(self.cubes.aux_grid_bin(cube_id=exchange.text("id")))

    def transect(self, exchange: HttpExchange) -> None:
        png = self.cubes.transect_png(
            cube_id = exchange.text("id"),
            source  = exchange.text("source", "pred"),
            row0    = exchange.integer("row0"),
            col0    = exchange.integer("col0"),
            row1    = exchange.integer("row1"),
            col1    = exchange.integer("col1"),
            space   = exchange.text("space", "physical"),
            cmap    = exchange.text("cmap", "jet"),
        )
        exchange.send_png(png)

    def param_map(self, exchange: HttpExchange) -> None:
        png = self.cubes.param_map_png(
            cube_id = exchange.text("id"),
            source  = exchange.text("source", "pred"),
            field   = exchange.text("field", "amp"),
            slot    = exchange.integer("slot"),
        )
        exchange.send_png(png)

    def param_cbar(self, exchange: HttpExchange) -> None:
        png = self.cubes.param_cbar_png(
            cube_id = exchange.text("id"),
            source  = exchange.text("source", "pred"),
            field   = exchange.text("field", "amp"),
        )
        exchange.send_png(png)

    def params_at(self, exchange: HttpExchange) -> None:
        result = self.cubes.params_at(
            cube_id = exchange.text("id"),
            row     = exchange.integer("row"),
            col     = exchange.integer("col"),
        )
        exchange.send_result(result, 404)

    def slice_view(self, exchange: HttpExchange) -> None:
        png = self.cubes.slice_png(
            cube_id = exchange.text("id"),
            source  = exchange.text("source", "pred"),
            axis    = exchange.text("axis", "col"),
            row     = exchange.integer("row"),
            col     = exchange.integer("col"),
            space   = exchange.text("space", "physical"),
            cmap    = exchange.text("cmap", "jet"),
        )
        exchange.send_png(png)

    def load(self, exchange: HttpExchange) -> None:
        exchange.send_result(self.cubes.start_load(exchange.body.get("id", "")))

    def attach(self, exchange: HttpExchange) -> None:
        exchange.send_result(self.cubes.attach_second(exchange.body.get("id", ""), exchange.body.get("other", "")))

    def detach(self, exchange: HttpExchange) -> None:
        exchange.send_result(self.cubes.detach_second(exchange.body.get("id", "")))

    def save_transect(self, exchange: HttpExchange) -> None:
        body   = exchange.body
        result = self.cubes.save_transect(
            cube_id = body.get("id", ""),
            row0    = int(body.get("row0", 0)),
            col0    = int(body.get("col0", 0)),
            row1    = int(body.get("row1", 0)),
            col1    = int(body.get("col1", 0)),
            space   = body.get("space", "physical"),
            cmap    = body.get("cmap", "jet"),
        )
        exchange.send_result(result)

    def save_slices(self, exchange: HttpExchange) -> None:
        body   = exchange.body
        result = self.cubes.save_slices(
            cube_id = body.get("id", ""),
            row     = int(body.get("row", 0)),
            col     = int(body.get("col", 0)),
            space   = body.get("space", "physical"),
            cmap    = body.get("cmap", "jet"),
        )
        exchange.send_result(result)


class SliceRouter(SubRouter):
    """Routes the /api/slices endpoints onto the multi-run slice collector and its mailer.

    Attributes:
        slices : Slice collector service answering the requests.
        mailer : Service mailing a rendered basket.
    """

    def __init__(self, slices: SliceCollector, mailer: SliceCollectionMailer) -> None:
        """Stores the slice collector and mailer services and declares their routes."""
        self.slices = slices
        self.mailer = mailer

        super().__init__(("/api/slices",))

    def declare(self, table: RouteTable) -> None:
        """Registers the /api/slices routes."""
        table.add("GET",  "/api/slices/info",          self.info)
        table.add("GET",  "/api/slices/primary",       self.primary)
        table.add("GET",  "/api/slices/slice",         self.slice_view)
        table.add("GET",  "/api/slices/mail_settings", self.mail_settings)
        table.add("POST", "/api/slices/collect",       self.collect)
        table.add("POST", "/api/slices/mail",          self.mail)

    def info(self, exchange: HttpExchange) -> None:
        """Answers with the axis extents and sources available for one cube."""
        exchange.send_result(self.slices.info(exchange.text("id")), 404)

    def primary(self, exchange: HttpExchange) -> None:
        """Answers with the PNG primary-amplitude map of one cube, cropped to its region."""
        exchange.send_png(self.slices.primary_png(exchange.text("id")))

    def slice_view(self, exchange: HttpExchange) -> None:
        """Answers with the PNG slice of one cube under an optional shared colour scale."""
        png = self.slices.slice_png(
            cube_id = exchange.text("id"),
            source  = exchange.text("source", "pred"),
            axis    = exchange.text("axis", "col"),
            row     = exchange.integer("row"),
            col     = exchange.integer("col"),
            space   = exchange.text("space", "physical"),
            cmap    = exchange.text("cmap", "jet"),
            vmin    = exchange.optional_number("vmin"),
            vmax    = exchange.optional_number("vmax"),
        )
        exchange.send_png(png)

    def mail_settings(self, exchange: HttpExchange) -> None:
        """Answers with the default recipient and whether the SMTP password is available."""
        exchange.send_json(self.mailer.settings())

    def collect(self, exchange: HttpExchange) -> None:
        """Renders the posted basket of slices into one saved figure set."""
        body   = exchange.body
        result = self.slices.collect(
            items  = body.get("items") or [],
            shared = bool(body.get("shared", True)),
            name   = body.get("name", ""),
        )
        exchange.send_result(result)

    def mail(self, exchange: HttpExchange) -> None:
        """Renders the posted basket of slices, then mails it as zip attachments."""
        body   = exchange.body
        result = self.mailer.mail(
            items     = body.get("items") or [],
            shared    = bool(body.get("shared", True)),
            name      = body.get("name", ""),
            recipient = body.get("recipient", ""),
            note      = body.get("note", ""),
        )
        exchange.send_result(result)
