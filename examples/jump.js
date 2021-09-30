const fs = require('fs')
const { plot, stack } = require('nodeplotlib')
const { Physics, PlayerState } = require('prismarine-physics')
const { Vec3 } = require('vec3')
const RL = require('../lib/reinforcejs/rl')

const version = '1.16.5'

const mcData = require('minecraft-data')(version)
const Block = require('prismarine-block')(version)

const fakeWorld = require('../lib/world').getRandomWorld(mcData, Block)
const { fakePlayer } = require('../lib/player')

const sideViewData = [{ x: [], y: [], type: 'line', name: 'pos' }, { x: [], y: [], type: 'line', name: 'ground' }]
const posPlotData = [{ x: [], y: [], type: 'line', name: 'posX' }, { x: [], y: [], type: 'line', name: 'posY' }, { x: [], y: [], type: 'line', name: 'posZ' }]
const velPlotData = [{ x: [], y: [], type: 'line', name: 'velX' }, { x: [], y: [], type: 'line', name: 'velY' }, { x: [], y: [], type: 'line', name: 'velZ' }]
const DQNData = [{ x: [], y: [], type: 'line', name: 'reward' }]

// Draw the world
function addCube (data, x, y) {
  data.x.push(x, x + 1, x + 1, x, x, null)
  data.y.push(y, y, y + 1, y + 1, y, null)
}
for (const block of fakeWorld.blocks) {
  addCube(sideViewData[1], block.z, block.y)
}

function plotState (state, tick) {
  sideViewData[0].x.push(state.pos.z)
  sideViewData[0].y.push(state.pos.y)

  posPlotData[0].x.push(tick)
  posPlotData[0].y.push(state.pos.x)
  posPlotData[1].x.push(tick)
  posPlotData[1].y.push(state.pos.y)
  posPlotData[2].x.push(tick)
  posPlotData[2].y.push(state.pos.z)

  velPlotData[0].x.push(tick)
  velPlotData[0].y.push(state.vel.x)
  velPlotData[1].x.push(tick)
  velPlotData[1].y.push(state.vel.y)
  velPlotData[2].x.push(tick)
  velPlotData[2].y.push(state.vel.z)
}

function normxz (vec) {
  return Math.sqrt(vec.x * vec.x + vec.z * vec.z)
}

function dotxz (a, b) {
  return (a.x * b.x) + (a.z * b.z)
}

// Controller state
// Horizontal velocity
// Vertical velocity
// Vertical distance to ground
// Horizontal distance to edge
// Vertical distance to edge (positive or negative)
// Horizontal distance to point1
// Vertical distance to point1
// Horizontal distance from point1 to point2
// Vertical distance from point1 to point2
// Turn angle after point1
function getControllerState (path, playerState) {
  const hzVelocity = normxz(playerState.vel)
  const vtVelocity = playerState.vel.y

  const yaw = Math.PI - playerState.yaw
  const heading = new Vec3(-Math.sin(yaw), 0, Math.cos(yaw))

  const hzDistanceToPoint = path[0].xzDistanceTo(playerState.pos) * Math.sign(dotxz(heading, path[0].minus(playerState.pos)))
  const vtDistanceToPoint = path[0].y - playerState.pos.y

  const hzDistanceToPoint2 = path.length === 1 ? 0 : path[0].xzDistanceTo(path[1])
  const vtDistanceToPoint2 = path.length === 1 ? 0 : path[1].y - path[0].y

  return [hzVelocity, vtVelocity, hzDistanceToPoint, vtDistanceToPoint, hzDistanceToPoint2, vtDistanceToPoint2]
}

// Actions:
const actions = [
  // {forward: false, back: false, left: false, right: false, jump: false, sprint: false, sneak: false}, // none
  { forward: true, back: false, left: false, right: false, jump: false, sprint: false, sneak: false }, // forward
  // {forward: false, back: false, left: false, right: false, jump: true, sprint: false, sneak: false},  // jump
  { forward: true, back: false, left: false, right: false, jump: true, sprint: false, sneak: false }, // forward + jump
  // {forward: true, back: false, left: false, right: false, jump: false, sprint: true, sneak: false},   // forward + sprint
  // {forward: true, back: false, left: false, right: false, jump: true, sprint: true, sneak: false},    // forward + jump + sprint
  { forward: false, back: true, left: false, right: false, jump: false, sprint: false, sneak: false } // back
  // {forward: false, back: false, left: false, right: false, jump: false, sprint: false, sneak: true},  // sneak
  // {forward: true, back: false, left: false, right: false, jump: false, sprint: false, sneak: true}    // forward + sneak
]

// none, forward, jump, forward + jump, forward + sprint, forward + jump + sprint, back, sneak, forward + sneak

/* function bootstrapPolicy(s) {
  if (s[2] < 0) return actions[6]
  if (s[2] > 2 && s[2] < 5) return actions[5] // forward + jump + sprint
  if ((s[2] > 1 || s[3] > 0.6) && s[2] < 2) return actions[5] // forward + jump
  if (s[2] > 2) return actions[4] // forward + sprint
  return actions[1] // foward
} */

// Reward
function getReward (path, playerState) {
  const nextPoint = path[0]

  if (playerState.pos.y < nextPoint.y - 5) return -1 // penalize fall

  const d = nextPoint.xzDistanceTo(playerState.pos)
  if (d < 0.3 && playerState.onGround && Math.abs(playerState.pos.y - nextPoint.y) < 0.001) {
    path.shift()
    return 1 // - (d / 0.3)
  }

  return 0
}

const physics = Physics(mcData, fakeWorld)
let player = fakePlayer(new Vec3(0.5, 1, 0.5), version)
let playerState = new PlayerState(player, actions[0])
let path = [...fakeWorld.path]
let cumReward = 0
let epoch = 0
let tick = 0

const agent = new RL.DQNAgent({
  getNumStates: () => 4,
  getMaxNumActions: () => actions.length
}, { alpha: 0.01, epsilon: 0.2, gamma: 0.1, num_hidden_units: 100 })

// agent.fromJSON(require('./models/dqn.json'))
// agent.epsilon = 0 // deterministic

while (true) {
  if (path.length === 0) break
  const s = getControllerState(path, playerState)
  const action = actions[agent.act(s)]

  playerState.control = action
  physics.simulatePlayer(playerState, fakeWorld)
  tick++

  const reward = getReward(path, playerState)
  agent.learn(reward) // the agent improves its Q,policy,model, etc. reward is a float
  cumReward += reward

  if (reward === -1 || tick === 200) {
    console.log(`Reward: ${cumReward + 1}`, `Epoch: ${epoch}`)
    if (epoch === 200) break
    player = fakePlayer(new Vec3(0.5, 1, 0.5), version)
    playerState = new PlayerState(player, actions[0])
    path = [...fakeWorld.path]
    cumReward = 0
    tick = 0
    epoch++
  }
}

// save
fs.writeFileSync('./models/dqn.json', JSON.stringify(agent.toJSON(), null, 2), 'utf-8')

// plot result
player = fakePlayer(new Vec3(0.5, 1, 0.5), version)
playerState = new PlayerState(player, actions[0])
path = [...fakeWorld.path]
agent.epsilon = 0 // deterministic

plotState(playerState, 0)
for (let tick = 0; tick < 1000; tick++) {
  const s = getControllerState(path, playerState)
  const action = actions[agent.act(s)]

  playerState.control = action
  physics.simulatePlayer(playerState, fakeWorld).apply(player)
  const reward = getReward(path, playerState)

  DQNData[0].x.push(tick + 1)
  DQNData[0].y.push(reward)
  plotState(playerState, tick + 1)

  if (reward === -1) break
}

stack(sideViewData, {
  title: 'World YZ',
  xaxis: {
    constrain: 'domain'
  },
  yaxis: {
    scaleanchor: 'x'
  }
})
stack(posPlotData, { title: 'Position' })
stack(velPlotData, { title: 'Velocity' })
stack(DQNData, { title: 'Reward' })

plot()
