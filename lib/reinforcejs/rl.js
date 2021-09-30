const R = require('./recurrent')

// syntactic sugar function for getting default parameter values
const getopt = function (opt, fieldName, defaultValue) {
  if (typeof opt === 'undefined') { return defaultValue }
  return (typeof opt[fieldName] !== 'undefined') ? opt[fieldName] : defaultValue
}

const zeros = R.zeros // inherit these
const assert = R.assert
const randi = R.randi
// const randf = R.randf

const setConst = function (arr, c) {
  for (let i = 0, n = arr.length; i < n; i++) {
    arr[i] = c
  }
}

const sampleWeighted = function (p) {
  const r = Math.random()
  let c = 0.0
  for (let i = 0, n = p.length; i < n; i++) {
    c += p[i]
    if (c >= r) { return i }
  }
  assert(false, 'wtf')
}

// ------
// AGENTS
// ------

// DPAgent performs Value Iteration
// - can also be used for Policy Iteration if you really wanted to
// - requires model of the environment :(
// - does not learn from experience :(
// - assumes finite MDP :(
const DPAgent = function (env, opt) {
  this.V = null // state value function
  this.P = null // policy distribution \pi(s,a)
  this.env = env // store pointer to environment
  this.gamma = getopt(opt, 'gamma', 0.75) // future reward discount factor
  this.reset()
}
DPAgent.prototype = {
  reset: function () {
    // reset the agent's policy and value function
    this.ns = this.env.getNumStates()
    this.na = this.env.getMaxNumActions()
    this.V = zeros(this.ns)
    this.P = zeros(this.ns * this.na)
    // initialize uniform random policy
    for (let s = 0; s < this.ns; s++) {
      const poss = this.env.allowedActions(s)
      for (let i = 0, n = poss.length; i < n; i++) {
        this.P[poss[i] * this.ns + s] = 1.0 / poss.length
      }
    }
  },
  act: function (s) {
    // behave according to the learned policy
    const poss = this.env.allowedActions(s)
    const ps = []
    for (let i = 0, n = poss.length; i < n; i++) {
      const a = poss[i]
      const prob = this.P[a * this.ns + s]
      ps.push(prob)
    }
    const maxi = sampleWeighted(ps)
    return poss[maxi]
  },
  learn: function () {
    // perform a single round of value iteration
    this.evaluatePolicy() // writes this.V
    this.updatePolicy() // writes this.P
  },
  evaluatePolicy: function () {
    // perform a synchronous update of the value function
    const Vnew = zeros(this.ns)
    for (let s = 0; s < this.ns; s++) {
      // integrate over actions in a stochastic policy
      // note that we assume that policy probability mass over allowed actions sums to one
      let v = 0.0
      const poss = this.env.allowedActions(s)
      for (let i = 0, n = poss.length; i < n; i++) {
        const a = poss[i]
        const prob = this.P[a * this.ns + s] // probability of taking action under policy
        if (prob === 0) { continue } // no contribution, skip for speed
        const ns = this.env.nextStateDistribution(s, a)
        const rs = this.env.reward(s, a, ns) // reward for s->a->ns transition
        v += prob * (rs + this.gamma * this.V[ns])
      }
      Vnew[s] = v
    }
    this.V = Vnew // swap
  },
  updatePolicy: function () {
    // update policy to be greedy w.r.t. learned Value function
    for (let s = 0; s < this.ns; s++) {
      const poss = this.env.allowedActions(s)
      // compute value of taking each allowed action
      let vmax, nmax
      const vs = []
      for (let i = 0, n = poss.length; i < n; i++) {
        const a = poss[i]
        const ns = this.env.nextStateDistribution(s, a)
        const rs = this.env.reward(s, a, ns)
        const v = rs + this.gamma * this.V[ns]
        vs.push(v)
        if (i === 0 || v > vmax) { vmax = v; nmax = 1 } else if (v === vmax) { nmax += 1 }
      }
      // update policy smoothly across all argmaxy actions
      for (let i = 0, n = poss.length; i < n; i++) {
        const a = poss[i]
        this.P[a * this.ns + s] = (vs[i] === vmax) ? 1.0 / nmax : 0.0
      }
    }
  }
}

// QAgent uses TD (Q-Learning, SARSA)
// - does not require environment model :)
// - learns from experience :)
const TDAgent = function (env, opt) {
  this.update = getopt(opt, 'update', 'qlearn') // qlearn | sarsa
  this.gamma = getopt(opt, 'gamma', 0.75) // future reward discount factor
  this.epsilon = getopt(opt, 'epsilon', 0.1) // for epsilon-greedy policy
  this.alpha = getopt(opt, 'alpha', 0.01) // value function learning rate

  // class allows non-deterministic policy, and smoothly regressing towards the optimal policy based on Q
  this.smooth_policyUpdate = getopt(opt, 'smooth_policyUpdate', false)
  this.beta = getopt(opt, 'beta', 0.01) // learning rate for policy, if smooth updates are on

  // eligibility traces
  this.lambda = getopt(opt, 'lambda', 0) // eligibility trace decay. 0 = no eligibility traces used
  this.replacing_traces = getopt(opt, 'replacing_traces', true)

  // optional optimistic initial values
  this.q_initVal = getopt(opt, 'q_initVal', 0)

  this.planN = getopt(opt, 'planN', 0) // number of planning steps per learning iteration (0 = no planning)

  this.Q = null // state action value function
  this.P = null // policy distribution \pi(s,a)
  this.e = null // eligibility trace
  this.env_modelS = null // environment model (s,a) -> (s',r)
  this.env_model_r = null // environment model (s,a) -> (s',r)
  this.env = env // store pointer to environment
  this.reset()
}
TDAgent.prototype = {
  reset: function () {
    // reset the agent's policy and value function
    this.ns = this.env.getNumStates()
    this.na = this.env.getMaxNumActions()
    this.Q = zeros(this.ns * this.na)
    if (this.q_initVal !== 0) { setConst(this.Q, this.q_initVal) }
    this.P = zeros(this.ns * this.na)
    this.e = zeros(this.ns * this.na)

    // model/planning vars
    this.env_modelS = zeros(this.ns * this.na)
    setConst(this.env_modelS, -1) // init to -1 so we can test if we saw the state before
    this.env_model_r = zeros(this.ns * this.na)
    this.saSeen = []
    this.pq = zeros(this.ns * this.na)

    // initialize uniform random policy
    for (let s = 0; s < this.ns; s++) {
      const poss = this.env.allowedActions(s)
      for (let i = 0, n = poss.length; i < n; i++) {
        this.P[poss[i] * this.ns + s] = 1.0 / poss.length
      }
    }
    // agent memory, needed for streaming updates
    // (s0,a0,r0,s1,a1,r1,...)
    this.r0 = null
    this.s0 = null
    this.s1 = null
    this.a0 = null
    this.a1 = null
  },
  resetEpisode: function () {
    // an episode finished
  },
  act: function (s) {
    // act according to epsilon greedy policy
    const poss = this.env.allowedActions(s)
    const probs = []
    for (let i = 0, n = poss.length; i < n; i++) {
      probs.push(this.P[poss[i] * this.ns + s])
    }
    let a
    // epsilon greedy policy
    if (Math.random() < this.epsilon) {
      a = poss[randi(0, poss.length)] // random available action
      this.explored = true
    } else {
      a = poss[sampleWeighted(probs)]
      this.explored = false
    }
    // shift state memory
    this.s0 = this.s1
    this.a0 = this.a1
    this.s1 = s
    this.a1 = a
    return a
  },
  learn: function (r1) {
    // takes reward for previous action, which came from a call to act()
    if (!(this.r0 == null)) {
      this.learnFromTuple(this.s0, this.a0, this.r0, this.s1, this.a1, this.lambda)
      if (this.planN > 0) {
        this.updateModel(this.s0, this.a0, this.r0, this.s1)
        this.plan()
      }
    }
    this.r0 = r1 // store this for next update
  },
  updateModel: function (s0, a0, r0, s1) {
    // transition (s0,a0) -> (r0,s1) was observed. Update environment model
    const sa = a0 * this.ns + s0
    if (this.env_modelS[sa] === -1) {
      // first time we see this state action
      this.saSeen.push(a0 * this.ns + s0) // add as seen state
    }
    this.env_modelS[sa] = s1
    this.env_model_r[sa] = r0
  },
  plan: function () {
    // order the states based on current priority queue information
    const spq = []
    for (let i = 0, n = this.saSeen.length; i < n; i++) {
      const sa = this.saSeen[i]
      const sap = this.pq[sa]
      if (sap > 1e-5) { // gain a bit of efficiency
        spq.push({ sa: sa, p: sap })
      }
    }
    spq.sort(function (a, b) { return a.p < b.p ? 1 : -1 })

    // perform the updates
    const nsteps = Math.min(this.planN, spq.length)
    for (let k = 0; k < nsteps; k++) {
      // random exploration
      // var i = randi(0, this.saSeen.length); // pick random prev seen state action
      // var s0a0 = this.saSeen[i];
      const s0a0 = spq[k].sa
      this.pq[s0a0] = 0 // erase priority, since we're backing up this state
      const s0 = s0a0 % this.ns
      const a0 = Math.floor(s0a0 / this.ns)
      const r0 = this.env_model_r[s0a0]
      const s1 = this.env_modelS[s0a0]
      let a1 = -1 // not used for Q learning
      if (this.update === 'sarsa') {
        // generate random action?...
        const poss = this.env.allowedActions(s1)
        a1 = poss[randi(0, poss.length)]
      }
      this.learnFromTuple(s0, a0, r0, s1, a1, 0) // note lambda = 0 - shouldnt use eligibility trace here
    }
  },
  learnFromTuple: function (s0, a0, r0, s1, a1, lambda) {
    const sa = a0 * this.ns + s0

    // calculate the target for Q(s,a)
    let target
    if (this.update === 'qlearn') {
      // Q learning target is Q(s0,a0) = r0 + gamma * max_a Q[s1,a]
      const poss = this.env.allowedActions(s1)
      let qmax = 0
      for (let i = 0, n = poss.length; i < n; i++) {
        const s1a = poss[i] * this.ns + s1
        const qval = this.Q[s1a]
        if (i === 0 || qval > qmax) { qmax = qval }
      }
      target = r0 + this.gamma * qmax
    } else if (this.update === 'sarsa') {
      // SARSA target is Q(s0,a0) = r0 + gamma * Q[s1,a1]
      const s1a1 = a1 * this.ns + s1
      target = r0 + this.gamma * this.Q[s1a1]
    }

    if (lambda > 0) {
      // perform an eligibility trace update
      if (this.replacing_traces) {
        this.e[sa] = 1
      } else {
        this.e[sa] += 1
      }
      const edecay = lambda * this.gamma
      const stateUpdate = zeros(this.ns)
      for (let s = 0; s < this.ns; s++) {
        const poss = this.env.allowedActions(s)
        for (let i = 0; i < poss.length; i++) {
          const a = poss[i]
          const saloop = a * this.ns + s
          const esa = this.e[saloop]
          const update = this.alpha * esa * (target - this.Q[saloop])
          this.Q[saloop] += update
          this.updatePriority(s, a, update)
          this.e[saloop] *= edecay
          const u = Math.abs(update)
          if (u > stateUpdate[s]) { stateUpdate[s] = u }
        }
      }
      for (let s = 0; s < this.ns; s++) {
        if (stateUpdate[s] > 1e-5) { // save efficiency here
          this.updatePolicy(s)
        }
      }
      if (this.explored && this.update === 'qlearn') {
        // have to wipe the trace since q learning is off-policy :(
        this.e = zeros(this.ns * this.na)
      }
    } else {
      // simpler and faster update without eligibility trace
      // update Q[sa] towards it with some step size
      const update = this.alpha * (target - this.Q[sa])
      this.Q[sa] += update
      this.updatePriority(s0, a0, update)
      // update the policy to reflect the change (if appropriate)
      this.updatePolicy(s0)
    }
  },
  updatePriority: function (s, a, u) {
    // used in planning. Invoked when Q[sa] += update
    // we should find all states that lead to (s,a) and upgrade their priority
    // of being update in the next planning step
    u = Math.abs(u)
    if (u < 1e-5) { return } // for efficiency skip small updates
    if (this.planN === 0) { return } // there is no planning to be done, skip.
    for (let si = 0; si < this.ns; si++) {
      // note we are also iterating over impossible actions at all states,
      // but this should be okay because their env_modelS should simply be -1
      // as initialized, so they will never be predicted to point to any state
      // because they will never be observed, and hence never be added to the model
      for (let ai = 0; ai < this.na; ai++) {
        const siai = ai * this.ns + si
        if (this.env_modelS[siai] === s) {
          // this state leads to s, add it to priority queue
          this.pq[siai] += u
        }
      }
    }
  },
  updatePolicy: function (s) {
    const poss = this.env.allowedActions(s)
    // set policy at s to be the action that achieves max_a Q(s,a)
    // first find the maxy Q values
    let qmax, nmax
    const qs = []
    for (let i = 0, n = poss.length; i < n; i++) {
      const a = poss[i]
      const qval = this.Q[a * this.ns + s]
      qs.push(qval)
      if (i === 0 || qval > qmax) { qmax = qval; nmax = 1 } else if (qval === qmax) { nmax += 1 }
    }
    // now update the policy smoothly towards the argmaxy actions
    let psum = 0.0
    for (let i = 0, n = poss.length; i < n; i++) {
      const a = poss[i]
      const target = (qs[i] === qmax) ? 1.0 / nmax : 0.0
      const ix = a * this.ns + s
      if (this.smooth_policyUpdate) {
        // slightly hacky :p
        this.P[ix] += this.beta * (target - this.P[ix])
        psum += this.P[ix]
      } else {
        // set hard target
        this.P[ix] = target
      }
    }
    if (this.smooth_policyUpdate) {
      // renomalize P if we're using smooth policy updates
      for (let i = 0, n = poss.length; i < n; i++) {
        const a = poss[i]
        this.P[a * this.ns + s] /= psum
      }
    }
  }
}

const DQNAgent = function (env, opt) {
  this.gamma = getopt(opt, 'gamma', 0.75) // future reward discount factor
  this.epsilon = getopt(opt, 'epsilon', 0.1) // for epsilon-greedy policy
  this.alpha = getopt(opt, 'alpha', 0.01) // value function learning rate

  this.experience_add_every = getopt(opt, 'experience_add_every', 25) // number of time steps before we add another experience to replay memory
  this.experienceSize = getopt(opt, 'experienceSize', 5000) // size of experience replay
  this.learningSteps_per_iteration = getopt(opt, 'learningSteps_per_iteration', 10)
  this.tderror_clamp = getopt(opt, 'tderror_clamp', 1.0)

  this.num_hiddenUnits = getopt(opt, 'num_hiddenUnits', 100)

  this.env = env
  this.reset()
}
DQNAgent.prototype = {
  reset: function () {
    this.nh = this.num_hiddenUnits // number of hidden units
    this.ns = this.env.getNumStates()
    this.na = this.env.getMaxNumActions()

    // nets are hardcoded for now as key (str) -> Mat
    // not proud of this. better solution is to have a whole Net object
    // on top of Mats, but for now sticking with this

    this.net = {}
    this.net.W1 = new R.RandMat(this.nh, this.ns, 0, 0.01)
    this.net.b1 = new R.Mat(this.nh, 1, 0, 0.01)
    this.net.W2 = new R.RandMat(this.na, this.nh, 0, 0.01)
    this.net.b2 = new R.Mat(this.na, 1, 0, 0.01)

    this.exp = [] // experience
    this.expi = 0 // where to insert

    this.t = 0

    this.r0 = null
    this.s0 = null
    this.s1 = null
    this.a0 = null
    this.a1 = null

    this.tderror = 0 // for visualization only...
  },
  toJSON: function () {
    // save function
    const j = {}
    j.nh = this.nh
    j.ns = this.ns
    j.na = this.na
    j.net = R.netToJSON(this.net)
    return j
  },
  fromJSON: function (j) {
    // load function
    this.nh = j.nh
    this.ns = j.ns
    this.na = j.na
    this.net = R.netFromJSON(j.net)
  },
  forwardQ: function (net, s, needsBackprop) {
    const G = new R.Graph(needsBackprop)
    const a1mat = G.add(G.mul(net.W1, s), net.b1)
    const h1mat = G.tanh(a1mat)
    const a2mat = G.add(G.mul(net.W2, h1mat), net.b2)
    this.lastG = G // back this up. Kind of hacky isn't it
    return a2mat
  },
  act: function (slist) {
    // convert to a Mat column vector
    const s = new R.Mat(this.ns, 1)
    s.setFrom(slist)

    // epsilon greedy policy
    let a
    if (Math.random() < this.epsilon) {
      a = randi(0, this.na)
    } else {
      // greedy wrt Q function
      const amat = this.forwardQ(this.net, s, false)
      a = R.maxi(amat.w) // returns index of argmax action
    }

    // shift state memory
    this.s0 = this.s1
    this.a0 = this.a1
    this.s1 = s
    this.a1 = a

    return a
  },
  learn: function (r1) {
    // perform an update on Q function
    if (!(this.r0 == null) && this.alpha > 0) {
      // learn from this tuple to get a sense of how "surprising" it is to the agent
      const tderror = this.learnFromTuple(this.s0, this.a0, this.r0, this.s1, this.a1)
      this.tderror = tderror // a measure of surprise

      // decide if we should keep this experience in the replay
      if (this.t % this.experience_add_every === 0) {
        this.exp[this.expi] = [this.s0, this.a0, this.r0, this.s1, this.a1]
        this.expi += 1
        if (this.expi > this.experienceSize) { this.expi = 0 } // roll over when we run out
      }
      this.t += 1

      // sample some additional experience from replay memory and learn from it
      for (let k = 0; k < this.learningSteps_per_iteration; k++) {
        const ri = randi(0, this.exp.length) // todo: priority sweeps?
        const e = this.exp[ri]
        this.learnFromTuple(e[0], e[1], e[2], e[3], e[4])
      }
    }
    this.r0 = r1 // store for next update
  },
  learnFromTuple: function (s0, a0, r0, s1, a1) {
    // want: Q(s,a) = r + gamma * max_a' Q(s',a')

    // compute the target Q value
    const tmat = this.forwardQ(this.net, s1, false)
    const qmax = r0 + this.gamma * tmat.w[R.maxi(tmat.w)]

    // now predict
    const pred = this.forwardQ(this.net, s0, true)

    let tderror = pred.w[a0] - qmax
    const clamp = this.tderror_clamp
    if (Math.abs(tderror) > clamp) { // huber loss to robustify
      if (tderror > clamp) tderror = clamp
      if (tderror < -clamp) tderror = -clamp
    }
    pred.dw[a0] = tderror
    this.lastG.backward() // compute gradients on net params

    // update net
    R.updateNet(this.net, this.alpha)
    return tderror
  }
}

// buggy implementation, doesnt work...
const SimpleReinforceAgent = function (env, opt) {
  this.gamma = getopt(opt, 'gamma', 0.5) // future reward discount factor
  this.epsilon = getopt(opt, 'epsilon', 0.75) // for epsilon-greedy policy
  this.alpha = getopt(opt, 'alpha', 0.001) // actor net learning rate
  this.beta = getopt(opt, 'beta', 0.01) // baseline net learning rate
  this.env = env
  this.reset()
}
SimpleReinforceAgent.prototype = {
  reset: function () {
    this.ns = this.env.getNumStates()
    this.na = this.env.getMaxNumActions()
    this.nh = 100 // number of hidden units
    this.nhb = 100 // and also in the baseline lstm

    this.actorNet = {}
    this.actorNet.W1 = new R.RandMat(this.nh, this.ns, 0, 0.01)
    this.actorNet.b1 = new R.Mat(this.nh, 1, 0, 0.01)
    this.actorNet.W2 = new R.RandMat(this.na, this.nh, 0, 0.1)
    this.actorNet.b2 = new R.Mat(this.na, 1, 0, 0.01)
    this.actorOutputs = []
    this.actorGraphs = []
    this.actorActions = [] // sampled ones

    this.rewardHistory = []

    this.baselineNet = {}
    this.baselineNet.W1 = new R.RandMat(this.nhb, this.ns, 0, 0.01)
    this.baselineNet.b1 = new R.Mat(this.nhb, 1, 0, 0.01)
    this.baselineNet.W2 = new R.RandMat(this.na, this.nhb, 0, 0.01)
    this.baselineNet.b2 = new R.Mat(this.na, 1, 0, 0.01)
    this.baselineOutputs = []
    this.baselineGraphs = []

    this.t = 0
  },
  forwardActor: function (s, needsBackprop) {
    const net = this.actorNet
    const G = new R.Graph(needsBackprop)
    const a1mat = G.add(G.mul(net.W1, s), net.b1)
    const h1mat = G.tanh(a1mat)
    const a2mat = G.add(G.mul(net.W2, h1mat), net.b2)
    return { a: a2mat, G: G }
  },
  forwardValue: function (s, needsBackprop) {
    const net = this.baselineNet
    const G = new R.Graph(needsBackprop)
    const a1mat = G.add(G.mul(net.W1, s), net.b1)
    const h1mat = G.tanh(a1mat)
    const a2mat = G.add(G.mul(net.W2, h1mat), net.b2)
    return { a: a2mat, G: G }
  },
  act: function (slist) {
    // convert to a Mat column vector
    const s = new R.Mat(this.ns, 1)
    s.setFrom(slist)

    // forward the actor to get action output
    let ans = this.forwardActor(s, true)
    const amat = ans.a
    const ag = ans.G
    this.actorOutputs.push(amat)
    this.actorGraphs.push(ag)

    // forward the baseline estimator
    ans = this.forwardValue(s, true)
    const vmat = ans.a
    const vg = ans.G
    this.baselineOutputs.push(vmat)
    this.baselineGraphs.push(vg)

    // sample action from the stochastic gaussian policy
    const a = R.copyMat(amat)
    const gaussVar = 0.02
    a.w[0] = R.randn(0, gaussVar)
    a.w[1] = R.randn(0, gaussVar)

    this.actorActions.push(a)

    // shift state memory
    this.s0 = this.s1
    this.a0 = this.a1
    this.s1 = s
    this.a1 = a

    return a
  },
  learn: function (r1) {
    // perform an update on Q function
    this.rewardHistory.push(r1)
    const n = this.rewardHistory.length
    let baselineMSE = 0.0
    const nup = 100 // what chunk of experience to take
    const nuse = 80 // what chunk to update from
    if (n >= nup) {
      // lets learn and flush
      // first: compute the sample values at all points
      const vs = []
      for (let t = 0; t < nuse; t++) {
        let mul = 1
        // compute the actual discounted reward for this time step
        let V = 0
        for (let t2 = t; t2 < n; t2++) {
          V += mul * this.rewardHistory[t2]
          mul *= this.gamma
          if (mul < 1e-5) { break } // efficiency savings
        }
        // get the predicted baseline at this time step
        const b = this.baselineOutputs[t].w[0]
        for (let i = 0; i < this.na; i++) {
          // [the action delta] * [the desirebility]
          let update = -(V - b) * (this.actorActions[t].w[i] - this.actorOutputs[t].w[i])
          if (update > 0.1) { update = 0.1 }
          if (update < -0.1) { update = -0.1 }
          this.actorOutputs[t].dw[i] += update
        }
        let update = -(V - b)
        if (update > 0.1) { update = 0.1 }
        if (update < 0.1) { update = -0.1 }
        this.baselineOutputs[t].dw[0] += update
        baselineMSE += (V - b) * (V - b)
        vs.push(V)
      }
      baselineMSE /= nuse
      // backprop all the things
      for (let t = 0; t < nuse; t++) {
        this.actorGraphs[t].backward()
        this.baselineGraphs[t].backward()
      }
      R.updateNet(this.actorNet, this.alpha) // update actor network
      R.updateNet(this.baselineNet, this.beta) // update baseline network

      // flush
      this.actorOutputs = []
      this.rewardHistory = []
      this.actorActions = []
      this.baselineOutputs = []
      this.actorGraphs = []
      this.baselineGraphs = []

      this.tderror = baselineMSE
    }
    this.t += 1
    this.r0 = r1 // store for next update
  }
}

// buggy implementation as well, doesn't work
const RecurrentReinforceAgent = function (env, opt) {
  this.gamma = getopt(opt, 'gamma', 0.5) // future reward discount factor
  this.epsilon = getopt(opt, 'epsilon', 0.1) // for epsilon-greedy policy
  this.alpha = getopt(opt, 'alpha', 0.001) // actor net learning rate
  this.beta = getopt(opt, 'beta', 0.01) // baseline net learning rate
  this.env = env
  this.reset()
}
RecurrentReinforceAgent.prototype = {
  reset: function () {
    this.ns = this.env.getNumStates()
    this.na = this.env.getMaxNumActions()
    this.nh = 40 // number of hidden units
    this.nhb = 40 // and also in the baseline lstm

    this.actorLSTM = R.initLSTM(this.ns, [this.nh], this.na)
    this.actorG = new R.Graph()
    this.actorPrev = null
    this.actorOutputs = []
    this.rewardHistory = []
    this.actorActions = []

    this.baselineLSTM = R.initLSTM(this.ns, [this.nhb], 1)
    this.baselineG = new R.Graph()
    this.baselinePrev = null
    this.baselineOutputs = []

    this.t = 0

    this.r0 = null
    this.s0 = null
    this.s1 = null
    this.a0 = null
    this.a1 = null
  },
  act: function (slist) {
    // convert to a Mat column vector
    const s = new R.Mat(this.ns, 1)
    s.setFrom(slist)

    // forward the LSTM to get action distribution
    const actorNext = R.forwardLSTM(this.actorG, this.actorLSTM, [this.nh], s, this.actorPrev)
    this.actorPrev = actorNext
    const amat = actorNext.o
    this.actorOutputs.push(amat)

    // forward the baseline LSTM
    const baselineNext = R.forwardLSTM(this.baselineG, this.baselineLSTM, [this.nhb], s, this.baselinePrev)
    this.baselinePrev = baselineNext
    this.baselineOutputs.push(baselineNext.o)

    // sample action from actor policy
    const gaussVar = 0.05
    const a = R.copyMat(amat)
    for (let i = 0, n = a.w.length; i < n; i++) {
      a.w[0] += R.randn(0, gaussVar)
      a.w[1] += R.randn(0, gaussVar)
    }
    this.actorActions.push(a)

    // shift state memory
    this.s0 = this.s1
    this.a0 = this.a1
    this.s1 = s
    this.a1 = a
    return a
  },
  learn: function (r1) {
    // perform an update on Q function
    this.rewardHistory.push(r1)
    const n = this.rewardHistory.length
    let baselineMSE = 0.0
    const nup = 100 // what chunk of experience to take
    const nuse = 80 // what chunk to also update
    if (n >= nup) {
      // lets learn and flush
      // first: compute the sample values at all points
      const vs = []
      for (let t = 0; t < nuse; t++) {
        let mul = 1
        let V = 0
        for (let t2 = t; t2 < n; t2++) {
          V += mul * this.rewardHistory[t2]
          mul *= this.gamma
          if (mul < 1e-5) { break } // efficiency savings
        }
        const b = this.baselineOutputs[t].w[0]
        // todo: take out the constants etc.
        for (let i = 0; i < this.na; i++) {
          // [the action delta] * [the desirebility]
          let update = -(V - b) * (this.actorActions[t].w[i] - this.actorOutputs[t].w[i])
          if (update > 0.1) { update = 0.1 }
          if (update < -0.1) { update = -0.1 }
          this.actorOutputs[t].dw[i] += update
        }
        let update = -(V - b)
        if (update > 0.1) { update = 0.1 }
        if (update < 0.1) { update = -0.1 }
        this.baselineOutputs[t].dw[0] += update
        baselineMSE += (V - b) * (V - b)
        vs.push(V)
      }
      baselineMSE /= nuse
      this.actorG.backward() // update params! woohoo!
      this.baselineG.backward()
      R.updateNet(this.actorLSTM, this.alpha) // update actor network
      R.updateNet(this.baselineLSTM, this.beta) // update baseline network

      // flush
      this.actorG = new R.Graph()
      this.actorPrev = null
      this.actorOutputs = []
      this.rewardHistory = []
      this.actorActions = []

      this.baselineG = new R.Graph()
      this.baselinePrev = null
      this.baselineOutputs = []

      this.tderror = baselineMSE
    }
    this.t += 1
    this.r0 = r1 // store for next update
  }
}

// Currently buggy implementation, doesnt work
const DeterministPG = function (env, opt) {
  this.gamma = getopt(opt, 'gamma', 0.5) // future reward discount factor
  this.epsilon = getopt(opt, 'epsilon', 0.5) // for epsilon-greedy policy
  this.alpha = getopt(opt, 'alpha', 0.001) // actor net learning rate
  this.beta = getopt(opt, 'beta', 0.01) // baseline net learning rate
  this.env = env
  this.reset()
}
DeterministPG.prototype = {
  reset: function () {
    this.ns = this.env.getNumStates()
    this.na = this.env.getMaxNumActions()
    this.nh = 100 // number of hidden units

    // actor
    this.actorNet = {}
    this.actorNet.W1 = new R.RandMat(this.nh, this.ns, 0, 0.01)
    this.actorNet.b1 = new R.Mat(this.nh, 1, 0, 0.01)
    this.actorNet.W2 = new R.RandMat(this.na, this.ns, 0, 0.1)
    this.actorNet.b2 = new R.Mat(this.na, 1, 0, 0.01)
    this.ntheta = this.na * this.ns + this.na // number of params in actor

    // critic
    this.criticw = new R.RandMat(1, this.ntheta, 0, 0.01) // row vector

    this.r0 = null
    this.s0 = null
    this.s1 = null
    this.a0 = null
    this.a1 = null
    this.t = 0
  },
  forwardActor: function (s, needsBackprop) {
    const net = this.actorNet
    const G = new R.Graph(needsBackprop)
    const a1mat = G.add(G.mul(net.W1, s), net.b1)
    const h1mat = G.tanh(a1mat)
    const a2mat = G.add(G.mul(net.W2, h1mat), net.b2)
    return { a: a2mat, G: G }
  },
  act: function (slist) {
    // convert to a Mat column vector
    const s = new R.Mat(this.ns, 1)
    s.setFrom(slist)

    // forward the actor to get action output
    const ans = this.forwardActor(s, false)
    const amat = ans.a

    // sample action from the stochastic gaussian policy
    const a = R.copyMat(amat)
    if (Math.random() < this.epsilon) {
      const gaussVar = 0.02
      a.w[0] = R.randn(0, gaussVar)
      a.w[1] = R.randn(0, gaussVar)
    }
    const clamp = 0.25
    if (a.w[0] > clamp) a.w[0] = clamp
    if (a.w[0] < -clamp) a.w[0] = -clamp
    if (a.w[1] > clamp) a.w[1] = clamp
    if (a.w[1] < -clamp) a.w[1] = -clamp

    // shift state memory
    this.s0 = this.s1
    this.a0 = this.a1
    this.s1 = s
    this.a1 = a

    return a
  },
  utilJacobianAt: function (s) {
    const ujacobian = new R.Mat(this.ntheta, this.na)
    for (let a = 0; a < this.na; a++) {
      R.netZeroGrads(this.actorNet)
      const ag = this.forwardActor(this.s0, true)
      ag.a.dw[a] = 1.0
      ag.G.backward()
      const gflat = R.netFlattenGrads(this.actorNet)
      ujacobian.setColumn(gflat, a)
    }
    return ujacobian
  },
  learn: function (r1) {
    // perform an update on Q function
    // this.rewardHistory.push(r1);
    if (!(this.r0 == null)) {
      const Gtmp = new R.Graph(false)
      // dpg update:
      // first compute the features psi:
      // the jacobian matrix of the actor for s
      const ujacobian0 = this.utilJacobianAt(this.s0)
      // now form the features \psi(s,a)
      const psiSa0 = Gtmp.mul(ujacobian0, this.a0) // should be [this.ntheta x 1] "feature" vector
      const qw0 = Gtmp.mul(this.criticw, psiSa0) // 1x1
      // now do the same thing because we need \psi(s_{t+1}, \mu\_\theta(s\_t{t+1}))
      const ujacobian1 = this.utilJacobianAt(this.s1)
      const ag = this.forwardActor(this.s1, false)
      const psiSa1 = Gtmp.mul(ujacobian1, ag.a)
      const qw1 = Gtmp.mul(this.criticw, psiSa1) // 1x1
      // get the td error finally
      let tderror = this.r0 + this.gamma * qw1.w[0] - qw0.w[0] // lol
      if (tderror > 0.5) tderror = 0.5 // clamp
      if (tderror < -0.5) tderror = -0.5
      this.tderror = tderror

      // update actor policy with natural gradient
      const net = this.actorNet
      let ix = 0
      for (const p in net) {
        const mat = net[p]
        if (Object.prototype.hasOwnProperty.call(net, p)) {
          for (let i = 0, n = mat.w.length; i < n; i++) {
            mat.w[i] += this.alpha * this.criticw.w[ix] // natural gradient update
            ix += 1
          }
        }
      }
      // update the critic parameters too
      for (let i = 0; i < this.ntheta; i++) {
        const update = this.beta * tderror * psiSa0.w[i]
        this.criticw.w[i] += update
      }
    }
    this.r0 = r1 // store for next update
  }
}

// exports
module.exports = {
  DPAgent,
  TDAgent,
  DQNAgent
  // global.SimpleReinforceAgent = SimpleReinforceAgent;
  // global.RecurrentReinforceAgent = RecurrentReinforceAgent;
  // global.DeterministPG = DeterministPG;
}
