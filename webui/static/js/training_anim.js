"use strict";

class TrainingScene extends CanvasBase {

  constructor(canvas) {
    super(canvas);
    this.t     = 0;
    this.frame = 0;
    this.blue  = "29, 79, 216";
    this.teal  = "15, 118, 110";
    this.ink   = "20, 30, 40";

    this.nLanes    = 4;
    this.lanes     = [];
    this.batches   = [];
    this.history   = [];
    this.steps     = 0;
    this.epoch     = 0;
    this.loss      = 1.0;
    this.noise     = 0;
    this.maxPoints = 220;

    this._seed();

    if (REDUCED_MOTION) {
      this._draw();
    } else {
      this._loop = this._loop.bind(this);
      requestAnimationFrame(this._loop);
    }
  }

  onResize() {
    if (!this.history) return;

    this._seed();
    if (REDUCED_MOTION) this._draw();
  }

  _designH() {
    return Math.min(this.h, window.innerHeight || this.h);
  }

  _laneY(i) {
    return this._designH() * (0.12 + 0.05 * i);
  }

  _seed() {
    this.lanes = [];
    for (let i = 0; i < this.nLanes; i++) {
      this.lanes.push({ lane: i, speed: 0.32 + i * 0.05, offset: Math.random() * (this.w + 120) });
    }

    this.batches = [];
    for (let i = 0; i < 14; i++) {
      this.batches.push({
        lane   : i % this.nLanes,
        at     : Math.random(),
        speed  : 0.055 + Math.random() * 0.05,
        weight : 0.35 + Math.random() * 0.65,
      });
    }

    if (!this.history.length) {
      for (let i = 0; i < 150; i++) this._step();
    }
  }

  _step() {
    this.noise  = this.noise * 0.86 + (Math.random() - 0.5) * 0.09;
    this.loss   = Math.max(0.035, this.loss * 0.985 + this.noise * 0.02);
    this.steps += 1;

    this.history.push(this.loss);
    if (this.history.length > this.maxPoints) this.history.shift();

    if (this.steps % 44 === 0) this.epoch += 1;

    if (this.loss < 0.05 || this.history.length === this.maxPoints) {
      this.loss    = 0.9 + Math.random() * 0.2;
      this.noise   = 0;
      this.steps   = 0;
      this.epoch   = 0;
      this.history = [];
    }
  }

  _drawGrid(ctx) {
    const dh = this._designH();

    ctx.save();
    ctx.setLineDash([3, 9]);
    ctx.lineWidth   = 1;
    ctx.strokeStyle = `rgba(${this.ink}, 0.07)`;
    for (let i = 0; i <= 5; i++) {
      const y = dh * (0.24 + 0.132 * i);
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(this.w, y);
      ctx.stroke();
    }
    ctx.restore();
  }

  _drawLanes(ctx) {
    ctx.save();
    ctx.setLineDash([2, 10]);
    ctx.lineWidth   = 1;
    ctx.strokeStyle = `rgba(${this.blue}, 0.10)`;
    for (let i = 0; i < this.nLanes; i++) {
      const y = this._laneY(i);
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(this.w * 0.9, y);
      ctx.stroke();
    }
    ctx.restore();

    for (const batch of this.batches) {
      const x = batch.at * this.w * 0.9;
      const y = this._laneY(batch.lane);
      const r = 1.4 + batch.weight * 2.2;

      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${this.blue}, ${0.18 + 0.32 * batch.weight})`;
      ctx.fill();

      ctx.beginPath();
      ctx.moveTo(x - r - 9, y);
      ctx.lineTo(x - r - 2, y);
      ctx.strokeStyle = `rgba(${this.blue}, ${0.10 * batch.weight})`;
      ctx.lineWidth   = 1;
      ctx.stroke();
    }
  }

  _drawStack(ctx) {
    const dh = this._designH();
    const x  = this.w * 0.9;
    const y0 = this._laneY(0) - 10;
    const y1 = this._laneY(this.nLanes - 1) + 10;

    ctx.beginPath();
    ctx.rect(x, y0, this.w * 0.045, y1 - y0);
    ctx.strokeStyle = `rgba(${this.ink}, 0.16)`;
    ctx.lineWidth   = 1.1;
    ctx.stroke();

    for (let i = 1; i < 4; i++) {
      const y = y0 + ((y1 - y0) * i) / 4;
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x + this.w * 0.045, y);
      ctx.strokeStyle = `rgba(${this.ink}, 0.08)`;
      ctx.stroke();
    }

    ctx.font         = "600 9px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.textAlign    = "left";
    ctx.textBaseline = "bottom";
    ctx.fillStyle    = `rgba(${this.ink}, 0.28)`;
    ctx.fillText("forward / backward", x, y0 - 7);
    ctx.fillText(`epoch ${this.epoch}`, this.w * 0.045, dh * 0.94);
  }

  _drawCurve(ctx) {
    if (this.history.length < 2) return;

    const dh   = this._designH();
    const x0   = this.w * 0.045;
    const x1   = this.w * 0.86;
    const yTop = dh * 0.30;
    const yBot = dh * 0.88;
    const span = x1 - x0;

    ctx.beginPath();
    ctx.moveTo(x0, yTop);
    ctx.lineTo(x0, yBot);
    ctx.lineTo(x1, yBot);
    ctx.strokeStyle = `rgba(${this.ink}, 0.14)`;
    ctx.lineWidth   = 1;
    ctx.stroke();

    const point = (i) => {
      const f = i / (this.maxPoints - 1);
      return [x0 + span * f, yBot - (yBot - yTop) * Math.min(1, this.history[i])];
    };

    ctx.beginPath();
    for (let i = 0; i < this.history.length; i++) {
      const [px, py] = point(i);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.strokeStyle = `rgba(${this.teal}, 0.45)`;
    ctx.lineWidth   = 1.4;
    ctx.stroke();

    const [lastX, lastY] = point(this.history.length - 1);
    ctx.lineTo(lastX, yBot);
    ctx.lineTo(x0, yBot);
    ctx.closePath();
    ctx.fillStyle = `rgba(${this.teal}, 0.06)`;
    ctx.fill();

    ctx.beginPath();
    ctx.arc(lastX, lastY, 2.6, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(${this.teal}, 0.55)`;
    ctx.fill();

    ctx.font         = "600 9px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.textAlign    = "left";
    ctx.textBaseline = "bottom";
    ctx.fillStyle    = `rgba(${this.ink}, 0.30)`;
    ctx.fillText("training loss", x0, yTop - 8);
  }

  _draw() {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.w, this.h);

    this._drawGrid(ctx);
    this._drawLanes(ctx);
    this._drawStack(ctx);
    this._drawCurve(ctx);
  }

  _advance() {
    for (const batch of this.batches) {
      batch.at += batch.speed * 0.016;
      if (batch.at > 1) {
        batch.at     = 0;
        batch.lane   = Math.floor(Math.random() * this.nLanes);
        batch.weight = 0.35 + Math.random() * 0.65;
      }
    }
  }

  _loop() {
    this.t     += 0.016;
    this.frame += 1;

    this._advance();
    if (this.frame % 6 === 0) this._step();

    requestAnimationFrame(this._loop);
    this._draw();
  }
}

window.TrainingScene = TrainingScene;
