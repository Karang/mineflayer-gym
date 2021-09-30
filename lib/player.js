const { Vec3 } = require('vec3')

function fakePlayer (pos, version) {
  return {
    entity: {
      position: pos,
      velocity: new Vec3(0, 0, 0),
      onGround: false,
      isInWater: false,
      isInLava: false,
      isInWeb: false,
      isCollidedHorizontally: false,
      isCollidedVertically: false,
      yaw: Math.PI,
      effects: {}
    },
    jumpTicks: 0,
    jumpQueued: false,
    version,
    inventory: {
      slots: []
    }
  }
}

module.exports = { fakePlayer }
