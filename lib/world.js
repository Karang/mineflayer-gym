const { Vec3 } = require('vec3')
const seedrandom = require('seedrandom')

const getRandomWorld = (mcData, Block, seed = 4242, maxLength = 10, maxGap = 1) => {
  const rng = seedrandom(seed)

  // Generate blocks
  const blocks = [new Vec3(0, 0, 0)]
  for (let i = 0; i < 20; i++) {
    const length = Math.floor(rng() * (maxLength + 1))
    const gap = Math.floor(rng() * (maxGap + 1))
    const Yoffset = Math.floor(rng() * 3) - 1
    let pos = blocks[blocks.length - 1]
    for (let i = 0; i < length; i++) {
      pos = pos.offset(0, Yoffset, 1 + gap)
      blocks.push(pos)
    }
  }

  // Generate path
  const path = []
  for (const b of blocks) {
    path.push(b.offset(0.5, 1, 0.5))
  }

  const bset = new Set(blocks.map(b => b + ''))
  // Fake world object
  return {
    getBlock: (pos) => {
      let type = mcData.blocksByName.air.id
      if (bset.has(pos + '')) type = mcData.blocksByName.stone.id
      const b = new Block(type, 0, 0)
      b.position = pos
      return b
    },
    blocks,
    path
  }
}

module.exports = { getRandomWorld }
