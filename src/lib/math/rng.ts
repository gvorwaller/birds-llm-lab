export interface RandomState {
  readonly state: number;
  readonly spareNormal: number | null;
}

const UINT32_RANGE = 4_294_967_296;

export class SeededRandom {
  private state: number;
  private spareNormal: number | null;

  constructor(seedOrState: number | RandomState) {
    if (typeof seedOrState === 'number') {
      if (!Number.isSafeInteger(seedOrState)) {
        throw new Error('Random seed must be a safe integer.');
      }
      this.state = seedOrState >>> 0;
      this.spareNormal = null;
    } else {
      if (
        !Number.isInteger(seedOrState.state) ||
        seedOrState.state < 0 ||
        seedOrState.state > 0xffff_ffff
      ) {
        throw new Error('Random state must be an unsigned 32-bit integer.');
      }
      if (seedOrState.spareNormal !== null && !Number.isFinite(seedOrState.spareNormal)) {
        throw new Error('Saved normal variate must be finite or null.');
      }
      this.state = seedOrState.state >>> 0;
      this.spareNormal = seedOrState.spareNormal;
    }
  }

  nextUint32(): number {
    this.state = (Math.imul(this.state, 1_664_525) + 1_013_904_223) >>> 0;
    return this.state;
  }

  uniform(): number {
    return this.nextUint32() / UINT32_RANGE;
  }

  normal(): number {
    if (this.spareNormal !== null) {
      const value = this.spareNormal;
      this.spareNormal = null;
      return value;
    }
    const radius = Math.sqrt(-2 * Math.log(1 - this.uniform()));
    const angle = 2 * Math.PI * this.uniform();
    this.spareNormal = radius * Math.sin(angle);
    return radius * Math.cos(angle);
  }

  snapshot(): RandomState {
    return { state: this.state, spareNormal: this.spareNormal };
  }
}
