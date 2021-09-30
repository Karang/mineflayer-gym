
// Utility fun
function assert (condition, message) {
  // from http://stackoverflow.com/questions/15313418/javascript-assert
  if (!condition) {
    message = message || 'Assertion failed'
    if (typeof Error !== 'undefined') {
      throw new Error(message)
    }
    throw message // Fallback
  }
}

// Random numbers utils
let returnV = false
let vVal = 0.0
const gaussRandom = function () {
  if (returnV) {
    returnV = false
    return vVal
  }
  const u = 2 * Math.random() - 1
  const v = 2 * Math.random() - 1
  const r = u * u + v * v
  if (r === 0 || r > 1) return gaussRandom()
  const c = Math.sqrt(-2 * Math.log(r) / r)
  vVal = v * c // cache this
  returnV = true
  return u * c
}
const randf = function (a, b) { return Math.random() * (b - a) + a }
const randi = function (a, b) { return Math.floor(Math.random() * (b - a) + a) }
const randn = function (mu, std) { return mu + gaussRandom() * std }

// helper function returns array of zeros of length n
// and uses typed arrays if available
const zeros = function (n) {
  if (typeof (n) === 'undefined' || isNaN(n)) { return [] }
  if (typeof ArrayBuffer === 'undefined') {
    // lacking browser support
    const arr = new Array(n)
    for (let i = 0; i < n; i++) { arr[i] = 0 }
    return arr
  } else {
    return new Float64Array(n)
  }
}

// Mat holds a matrix
const Mat = function (n, d) {
  // n is number of rows d is number of columns
  this.n = n
  this.d = d
  this.w = zeros(n * d)
  this.dw = zeros(n * d)
}
Mat.prototype = {
  get: function (row, col) {
    // slow but careful accessor function
    // we want row-major order
    const ix = (this.d * row) + col
    assert(ix >= 0 && ix < this.w.length)
    return this.w[ix]
  },
  set: function (row, col, v) {
    // slow but careful accessor function
    const ix = (this.d * row) + col
    assert(ix >= 0 && ix < this.w.length)
    this.w[ix] = v
  },
  setFrom: function (arr) {
    for (let i = 0, n = arr.length; i < n; i++) {
      this.w[i] = arr[i]
    }
  },
  setColumn: function (m, i) {
    for (let q = 0, n = m.w.length; q < n; q++) {
      this.w[(this.d * q) + i] = m.w[q]
    }
  },
  toJSON: function () {
    const json = {}
    json.n = this.n
    json.d = this.d
    json.w = this.w
    return json
  },
  fromJSON: function (json) {
    this.n = json.n
    this.d = json.d
    this.w = zeros(this.n * this.d)
    this.dw = zeros(this.n * this.d)
    for (let i = 0, n = this.n * this.d; i < n; i++) {
      this.w[i] = json.w[i] // copy over weights
    }
  }
}

const copyMat = function (b) {
  const a = new Mat(b.n, b.d)
  a.setFrom(b.w)
  return a
}

const copyNet = function (net) {
  // nets are (k,v) pairs with k = string key, v = Mat()
  const newNet = {}
  for (const p in net) {
    if (Object.prototype.hasOwnProperty.call(net, p)) {
      newNet[p] = copyMat(net[p])
    }
  }
  return newNet
}

const updateMat = function (m, alpha) {
  // updates in place
  for (let i = 0, n = m.n * m.d; i < n; i++) {
    if (m.dw[i] !== 0) {
      m.w[i] += -alpha * m.dw[i]
      m.dw[i] = 0
    }
  }
}

const updateNet = function (net, alpha) {
  for (const p in net) {
    if (Object.prototype.hasOwnProperty.call(net, p)) {
      updateMat(net[p], alpha)
    }
  }
}

const netToJSON = function (net) {
  const j = {}
  for (const p in net) {
    if (Object.prototype.hasOwnProperty.call(net, p)) {
      j[p] = net[p].toJSON()
    }
  }
  return j
}
const netFromJSON = function (j) {
  const net = {}
  for (const p in j) {
    if (Object.prototype.hasOwnProperty.call(j, p)) {
      net[p] = new Mat(1, 1) // not proud of this
      net[p].fromJSON(j[p])
    }
  }
  return net
}
const netZeroGrads = function (net) {
  for (const p in net) {
    if (Object.prototype.hasOwnProperty.call(net, p)) {
      const mat = net[p]
      gradFillConst(mat, 0)
    }
  }
}
const netFlattenGrads = function (net) {
  let n = 0
  for (const p in net) { if (Object.prototype.hasOwnProperty.call(net, p)) { const mat = net[p]; n += mat.dw.length } }
  const g = new Mat(n, 1)
  let ix = 0
  for (const p in net) {
    if (Object.prototype.hasOwnProperty.call(net, p)) {
      const mat = net[p]
      for (let i = 0, m = mat.dw.length; i < m; i++) {
        g.w[ix] = mat.dw[i]
        ix++
      }
    }
  }
  return g
}

// return Mat but filled with random numbers from gaussian
const RandMat = function (n, d, mu, std) {
  const m = new Mat(n, d)
  fillRandn(m, mu, std)
  // fillRand(m,-std,std); // kind of :P
  return m
}

// Mat utils
// fill matrix with random gaussian numbers
const fillRandn = function (m, mu, std) { for (let i = 0, n = m.w.length; i < n; i++) { m.w[i] = randn(mu, std) } }
// const fillRand = function (m, lo, hi) { for (let i = 0, n = m.w.length; i < n; i++) { m.w[i] = randf(lo, hi) } }
const gradFillConst = function (m, c) { for (let i = 0, n = m.dw.length; i < n; i++) { m.dw[i] = c } }

// Transformer definitions
const Graph = function (needsBackprop) {
  if (typeof needsBackprop === 'undefined') { needsBackprop = true }
  this.needsBackprop = needsBackprop

  // this will store a list of functions that perform backprop,
  // in their forward pass order. So in backprop we will go
  // backwards and evoke each one
  this.backprop = []
}
Graph.prototype = {
  backward: function () {
    for (let i = this.backprop.length - 1; i >= 0; i--) {
      this.backprop[i]() // tick!
    }
  },
  rowPluck: function (m, ix) {
    // pluck a row of m with index ix and return it as col vector
    assert(ix >= 0 && ix < m.n)
    const d = m.d
    const out = new Mat(d, 1)
    for (let i = 0, n = d; i < n; i++) { out.w[i] = m.w[d * ix + i] } // copy over the data

    if (this.needsBackprop) {
      const backward = function () {
        for (let i = 0, n = d; i < n; i++) { m.dw[d * ix + i] += out.dw[i] }
      }
      this.backprop.push(backward)
    }
    return out
  },
  tanh: function (m) {
    // tanh nonlinearity
    const out = new Mat(m.n, m.d)
    const n = m.w.length
    for (let i = 0; i < n; i++) {
      out.w[i] = Math.tanh(m.w[i])
    }

    if (this.needsBackprop) {
      const backward = function () {
        for (let i = 0; i < n; i++) {
          // grad for z = tanh(x) is (1 - z^2)
          const mwi = out.w[i]
          m.dw[i] += (1.0 - mwi * mwi) * out.dw[i]
        }
      }
      this.backprop.push(backward)
    }
    return out
  },
  sigmoid: function (m) {
    // sigmoid nonlinearity
    const out = new Mat(m.n, m.d)
    const n = m.w.length
    for (let i = 0; i < n; i++) {
      out.w[i] = sig(m.w[i])
    }

    if (this.needsBackprop) {
      const backward = function () {
        for (let i = 0; i < n; i++) {
          // grad for z = tanh(x) is (1 - z^2)
          const mwi = out.w[i]
          m.dw[i] += mwi * (1.0 - mwi) * out.dw[i]
        }
      }
      this.backprop.push(backward)
    }
    return out
  },
  relu: function (m) {
    const out = new Mat(m.n, m.d)
    const n = m.w.length
    for (let i = 0; i < n; i++) {
      out.w[i] = Math.max(0, m.w[i]) // relu
    }
    if (this.needsBackprop) {
      const backward = function () {
        for (let i = 0; i < n; i++) {
          m.dw[i] += m.w[i] > 0 ? out.dw[i] : 0.0
        }
      }
      this.backprop.push(backward)
    }
    return out
  },
  mul: function (m1, m2) {
    // multiply matrices m1 * m2
    assert(m1.d === m2.n, 'matmul dimensions misaligned')

    const n = m1.n
    const d = m2.d
    const out = new Mat(n, d)
    for (let i = 0; i < m1.n; i++) { // loop over rows of m1
      for (let j = 0; j < m2.d; j++) { // loop over cols of m2
        let dot = 0.0
        for (let k = 0; k < m1.d; k++) { // dot product loop
          dot += m1.w[m1.d * i + k] * m2.w[m2.d * k + j]
        }
        out.w[d * i + j] = dot
      }
    }

    if (this.needsBackprop) {
      const backward = function () {
        for (let i = 0; i < m1.n; i++) { // loop over rows of m1
          for (let j = 0; j < m2.d; j++) { // loop over cols of m2
            for (let k = 0; k < m1.d; k++) { // dot product loop
              const b = out.dw[d * i + j]
              m1.dw[m1.d * i + k] += m2.w[m2.d * k + j] * b
              m2.dw[m2.d * k + j] += m1.w[m1.d * i + k] * b
            }
          }
        }
      }
      this.backprop.push(backward)
    }
    return out
  },
  add: function (m1, m2) {
    assert(m1.w.length === m2.w.length)

    const out = new Mat(m1.n, m1.d)
    for (let i = 0, n = m1.w.length; i < n; i++) {
      out.w[i] = m1.w[i] + m2.w[i]
    }
    if (this.needsBackprop) {
      const backward = function () {
        for (let i = 0, n = m1.w.length; i < n; i++) {
          m1.dw[i] += out.dw[i]
          m2.dw[i] += out.dw[i]
        }
      }
      this.backprop.push(backward)
    }
    return out
  },
  dot: function (m1, m2) {
    // m1 m2 are both column vectors
    assert(m1.w.length === m2.w.length)
    const out = new Mat(1, 1)
    let dot = 0.0
    for (let i = 0, n = m1.w.length; i < n; i++) {
      dot += m1.w[i] * m2.w[i]
    }
    out.w[0] = dot
    if (this.needs_backprop) {
      const backward = function () {
        for (let i = 0, n = m1.w.length; i < n; i++) {
          m1.dw[i] += m2.w[i] * out.dw[0]
          m2.dw[i] += m1.w[i] * out.dw[0]
        }
      }
      this.backprop.push(backward)
    }
    return out
  },
  eltmul: function (m1, m2) {
    assert(m1.w.length === m2.w.length)

    const out = new Mat(m1.n, m1.d)
    for (let i = 0, n = m1.w.length; i < n; i++) {
      out.w[i] = m1.w[i] * m2.w[i]
    }
    if (this.needs_backprop) {
      const backward = function () {
        for (let i = 0, n = m1.w.length; i < n; i++) {
          m1.dw[i] += m2.w[i] * out.dw[i]
          m2.dw[i] += m1.w[i] * out.dw[i]
        }
      }
      this.backprop.push(backward)
    }
    return out
  }
}

const softmax = function (m) {
  const out = new Mat(m.n, m.d) // probability volume
  let maxval = -999999
  for (let i = 0, n = m.w.length; i < n; i++) { if (m.w[i] > maxval) maxval = m.w[i] }

  let s = 0.0
  for (let i = 0, n = m.w.length; i < n; i++) {
    out.w[i] = Math.exp(m.w[i] - maxval)
    s += out.w[i]
  }
  for (let i = 0, n = m.w.length; i < n; i++) { out.w[i] /= s }

  // no backward pass here needed
  // since we will use the computed probabilities outside
  // to set gradients directly on m
  return out
}

const Solver = function () {
  this.decayRate = 0.999
  this.smoothEps = 1e-8
  this.stepCache = {}
}
Solver.prototype = {
  step: function (model, stepSize, regc, clipval) {
    // perform parameter update
    const solverStats = {}
    let numClipped = 0
    let numTot = 0
    for (const k in model) {
      if (Object.prototype.hasOwnProperty.call(model, k)) {
        const m = model[k] // mat ref
        if (!(k in this.stepCache)) { this.stepCache[k] = new Mat(m.n, m.d) }
        const s = this.stepCache[k]
        for (let i = 0, n = m.w.length; i < n; i++) {
          // rmsprop adaptive learning rate
          let mdwi = m.dw[i]
          s.w[i] = s.w[i] * this.decayRate + (1.0 - this.decayRate) * mdwi * mdwi

          // gradient clip
          if (mdwi > clipval) {
            mdwi = clipval
            numClipped++
          }
          if (mdwi < -clipval) {
            mdwi = -clipval
            numClipped++
          }
          numTot++

          // update (and regularize)
          m.w[i] += -stepSize * mdwi / Math.sqrt(s.w[i] + this.smoothEps) - regc * m.w[i]
          m.dw[i] = 0 // reset gradients for next iteration
        }
      }
    }
    solverStats.ratioClipped = numClipped * 1.0 / numTot
    return solverStats
  }
}

const initLSTM = function (inputSize, hiddenSizes, outputSize) {
  // hidden size should be a list

  const model = {}
  for (let d = 0; d < hiddenSizes.length; d++) { // loop over depths
    const prevSize = d === 0 ? inputSize : hiddenSizes[d - 1]
    const hiddenSize = hiddenSizes[d]

    // gates parameters
    model['Wix' + d] = new RandMat(hiddenSize, prevSize, 0, 0.08)
    model['Wih' + d] = new RandMat(hiddenSize, hiddenSize, 0, 0.08)
    model['bi' + d] = new Mat(hiddenSize, 1)
    model['Wfx' + d] = new RandMat(hiddenSize, prevSize, 0, 0.08)
    model['Wfh' + d] = new RandMat(hiddenSize, hiddenSize, 0, 0.08)
    model['bf' + d] = new Mat(hiddenSize, 1)
    model['Wox' + d] = new RandMat(hiddenSize, prevSize, 0, 0.08)
    model['Woh' + d] = new RandMat(hiddenSize, hiddenSize, 0, 0.08)
    model['bo' + d] = new Mat(hiddenSize, 1)
    // cell write params
    model['Wcx' + d] = new RandMat(hiddenSize, prevSize, 0, 0.08)
    model['Wch' + d] = new RandMat(hiddenSize, hiddenSize, 0, 0.08)
    model['bc' + d] = new Mat(hiddenSize, 1)
  }
  // decoder params
  model.Whd = new RandMat(outputSize, hiddenSizes[hiddenSizes.length - 1], 0, 0.08)
  model.bd = new Mat(outputSize, 1)
  return model
}

const forwardLSTM = function (G, model, hiddenSizes, x, prev) {
  // forward prop for a single tick of LSTM
  // G is graph to append ops to
  // model contains LSTM parameters
  // x is 1D column vector with observation
  // prev is a struct containing hidden and cell
  // from previous iteration

  let hiddenPrevs = []
  let cellPrevs = []
  if (prev == null || typeof prev.h === 'undefined') {
    for (let d = 0; d < hiddenSizes.length; d++) {
      hiddenPrevs.push(new Mat(hiddenSizes[d], 1))
      cellPrevs.push(new Mat(hiddenSizes[d], 1))
    }
  } else {
    hiddenPrevs = prev.h
    cellPrevs = prev.c
  }

  const hidden = []
  const cell = []
  for (let d = 0; d < hiddenSizes.length; d++) {
    const inputVector = d === 0 ? x : hidden[d - 1]
    const hiddenPrev = hiddenPrevs[d]
    const cellPrev = cellPrevs[d]

    // input gate
    const h0 = G.mul(model['Wix' + d], inputVector)
    const h1 = G.mul(model['Wih' + d], hiddenPrev)
    const inputGate = G.sigmoid(G.add(G.add(h0, h1), model['bi' + d]))

    // forget gate
    const h2 = G.mul(model['Wfx' + d], inputVector)
    const h3 = G.mul(model['Wfh' + d], hiddenPrev)
    const forgetGate = G.sigmoid(G.add(G.add(h2, h3), model['bf' + d]))

    // output gate
    const h4 = G.mul(model['Wox' + d], inputVector)
    const h5 = G.mul(model['Woh' + d], hiddenPrev)
    const outputGate = G.sigmoid(G.add(G.add(h4, h5), model['bo' + d]))

    // write operation on cells
    const h6 = G.mul(model['Wcx' + d], inputVector)
    const h7 = G.mul(model['Wch' + d], hiddenPrev)
    const cellWrite = G.tanh(G.add(G.add(h6, h7), model['bc' + d]))

    // compute new cell activation
    const retainCell = G.eltmul(forgetGate, cellPrev) // what do we keep from cell
    const writeCell = G.eltmul(inputGate, cellWrite) // what do we write to cell
    const cellD = G.add(retainCell, writeCell) // new cell contents

    // compute hidden state as gated, saturated cell activations
    const hiddenD = G.eltmul(outputGate, G.tanh(cellD))

    hidden.push(hiddenD)
    cell.push(cellD)
  }

  // one decoder to outputs at end
  const output = G.add(G.mul(model.Whd, hidden[hidden.length - 1]), model.bd)

  // return cell memory, hidden representation and output
  return { h: hidden, c: cell, o: output }
}

const sig = function (x) {
  // helper function for computing sigmoid
  return 1.0 / (1 + Math.exp(-x))
}

const maxi = function (w) {
  // argmax of array w
  let maxv = w[0]
  let maxix = 0
  for (let i = 1, n = w.length; i < n; i++) {
    const v = w[i]
    if (v > maxv) {
      maxix = i
      maxv = v
    }
  }
  return maxix
}

const samplei = function (w) {
  // sample argmax from w, assuming w are
  // probabilities that sum to one
  const r = randf(0, 1)
  let x = 0.0
  let i = 0
  while (true) {
    x += w[i]
    if (x > r) { return i }
    i++
  }
  // return w.length - 1 // pretty sure we should never get here?
}

module.exports = {
  // constious utils
  assert,
  zeros,
  maxi,
  samplei,
  randi,
  randn,
  softmax,
  // classes
  Mat,
  RandMat,
  forwardLSTM,
  initLSTM,
  // more utils
  updateMat,
  updateNet,
  copyMat,
  copyNet,
  netToJSON,
  netFromJSON,
  netZeroGrads,
  netFlattenGrads,
  // optimization
  Solver,
  Graph
}
