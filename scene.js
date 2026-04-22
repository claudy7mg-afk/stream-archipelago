import * as THREE from 'three/webgpu'
import {
  uniform,
  float,
  vec2,
  vec4,
  color,
  uv,
  mix,
  pass,
  mrt,
  output,
  normalView,
  diffuseColor,
  velocity,
  add,
  directionToColor,
  colorToDirection,
  sample,
  metalness,
  roughness,
  positionWorld,
  fract,
  abs,
  max,
  step,
  convertToTexture,
} from 'three/tsl'
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js'
import { ssgi } from 'three/examples/jsm/tsl/display/SSGINode.js'
import { ssr } from 'three/examples/jsm/tsl/display/SSRNode.js'
import { traa } from 'three/examples/jsm/tsl/display/TRAANode.js'
import { gaussianBlur } from 'three/examples/jsm/tsl/display/GaussianBlurNode.js'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import Stats from 'stats-gl'
import * as easings from 'eases-jsnext'
import { WaterPlane } from './WaterPlane.js'


// ─── Params ─────────────────────────────────────────────────────────────────
const params = {
  fov: 50,
  cameraEase: 'quadInOut',
  cameraTransitionDuration: 1.25,
  blur: 0,
  sunColor: '#fff5e6',
  sunIntensity: 2.2,
  ambientColor: '#b8d4f0',
  ambientIntensity: 0.8,
  exposure: 1.15,
  fogEnabled: true,
  fogColor: '#dce8f5',
  fogDensity: 0.012,
  skyTopColor: '#a8d8ea',
  skyBottomColor: '#ffecd2',
  sunX: -20,
  sunY: 18,
  sunZ: 15,
  shadowEnabled: true,
  shadowRadius: 6,
  shadowBlurSamples: 16,
  shadowBias: -0.001,
  shadowNormalBias: 0.02,
  shadowMapSize: 1024,
  debug: false,
}

// ─── Scene ──────────────────────────────────────────────────────────────────
const scene = new THREE.Scene()
scene.fog = params.fogEnabled ? new THREE.FogExp2(params.fogColor, params.fogDensity) : null

const camera = new THREE.PerspectiveCamera(params.fov, innerWidth / innerHeight, 0.1, 500)
camera.position.set(0, 8, 18)
camera.lookAt(0, 0, 0)

const renderer = new THREE.WebGPURenderer({
  antialias: false,
  requiredLimits: { maxStorageBuffersInVertexStage: 2, maxColorAttachmentBytesPerSample: 64 },
})
renderer.setPixelRatio(1)
renderer.setSize(innerWidth, innerHeight)
renderer.shadowMap.enabled = true
renderer.shadowMap.type = THREE.VSMShadowMap
renderer.setClearColor(params.skyTopColor)
renderer.toneMapping = THREE.AgXToneMapping
renderer.toneMappingExposure = params.exposure
renderer.domElement.style.cssText = 'position:fixed;top:0;left:0;z-index:-1;'
document.body.appendChild(renderer.domElement)
await renderer.init()

const skyTopColorU = uniform(new THREE.Color(params.skyTopColor))

// ─── Post-Processing (SSGI + TRAA + SSR) ────────────────────────────────────
const scenePass = pass(scene, camera)
scenePass.setMRT(
  mrt({
    output: output,
    diffuseColor: diffuseColor,
    normal: directionToColor(normalView),
    velocity: velocity,
    metalrough: vec2(metalness, roughness),
  }),
)

const scenePassColor = scenePass.getTextureNode('output')
const scenePassDiffuse = scenePass.getTextureNode('diffuseColor')
const scenePassDepth = scenePass.getTextureNode('depth')
const scenePassNormal = scenePass.getTextureNode('normal')
const scenePassVelocity = scenePass.getTextureNode('velocity')
const scenePassMetalRough = scenePass.getTextureNode('metalrough')

const sceneNormal = sample((uvCoord) => colorToDirection(scenePassNormal.sample(uvCoord)))

const giPass = ssgi(scenePassColor, scenePassDepth, sceneNormal, camera)
giPass.sliceCount.value = 2
giPass.stepCount.value = 4
giPass.radius.value = 3
giPass.expFactor.value = 2
giPass.thickness.value = 0.09
giPass.backfaceLighting.value = 0
giPass.aoIntensity.value = 2.8
giPass.giIntensity.value = 16
giPass.useLinearThickness.value = false
giPass.useScreenSpaceSampling.value = true
giPass.useTemporalFiltering = true
giPass.giEnabled = true
giPass.aoEnabled = true

const gi = giPass.rgb
const ao = giPass.a

// ─── SSR ─────────────────────────────────────────────────────────────────────
const ssrPass = ssr(scenePassColor, scenePassDepth, sceneNormal, scenePassMetalRough.r, scenePassMetalRough.g)
ssrPass.quality.value = 0.4
ssrPass.blurQuality.value = 1
ssrPass.maxDistance.value = 60
ssrPass.opacity.value = 1
ssrPass.thickness.value = 0.03
ssrPass.enabled = true

const ssrMasked = mix(skyTopColorU.mul(scenePassMetalRough.r), ssrPass.rgb, ssrPass.a)

// ─── RenderPipeline ─────────────────────────────────────────────────────────
const renderPipeline = new THREE.RenderPipeline(renderer)

const compositeGiAoSsr = vec4(
  add(scenePassColor.rgb.mul(ao), scenePassDiffuse.rgb.mul(gi)).add(ssrMasked),
  scenePassColor.a,
)
const traaGiAoSsr = traa(compositeGiAoSsr, scenePassDepth, scenePassVelocity, camera)

const blurDirectionU = uniform(params.blur * 10)
const blurPass = gaussianBlur(traaGiAoSsr, blurDirectionU, 10)
blurPass.textureNode = convertToTexture(traaGiAoSsr)

// Start with blur bypassed (blur is 0)
let blurActive = params.blur > 0
renderPipeline.outputNode = blurActive ? blurPass : traaGiAoSsr
renderPipeline.needsUpdate = true

function setBlurActive(active) {
  if (active === blurActive) return
  blurActive = active
  renderPipeline.outputNode = active ? blurPass : traaGiAoSsr
  renderPipeline.needsUpdate = true
}

// ─── Debug ──────────────────────────────────────────────────────────────────
const debugOverlay = document.createElement('div')
debugOverlay.style.cssText = 'position:fixed;inset:0;z-index:1;display:none;'
document.body.appendChild(debugOverlay)

const controls = new OrbitControls(camera, debugOverlay)
controls.enableDamping = true
controls.target.set(0, 1, 0)
controls.maxPolarAngle = Math.PI * 0.55
controls.enabled = params.debug

const stats = new Stats({ trackGPU: false, trackCPT: false })
document.body.appendChild(stats.dom)
stats.init(renderer)

// ─── Lighting ───────────────────────────────────────────────────────────────
const sunLight = new THREE.DirectionalLight(params.sunColor, params.sunIntensity)
sunLight.position.set(params.sunX, params.sunY, params.sunZ)
sunLight.castShadow = params.shadowEnabled
sunLight.shadow.mapSize.set(params.shadowMapSize, params.shadowMapSize)
sunLight.shadow.camera.near = 0.1
sunLight.shadow.camera.far = 60
sunLight.shadow.camera.left = -22
sunLight.shadow.camera.right = 22
sunLight.shadow.camera.top = 22
sunLight.shadow.camera.bottom = -22
sunLight.shadow.radius = params.shadowRadius
sunLight.shadow.blurSamples = params.shadowBlurSamples
sunLight.shadow.bias = params.shadowBias
sunLight.shadow.normalBias = params.shadowNormalBias
scene.add(sunLight)

const ambientLight = new THREE.AmbientLight(params.ambientColor, params.ambientIntensity)
scene.add(ambientLight)

// Soft hemisphere light for extra fill
const hemiLight = new THREE.HemisphereLight('#c8e6ff', '#ffe8c8', 0.4)
scene.add(hemiLight)

// ─── Sky Gradient ───────────────────────────────────────────────────────────
const skyBottomColorU = uniform(new THREE.Color(params.skyBottomColor))
const skyHeight = 40
const skyGeo = new THREE.PlaneGeometry(160, skyHeight)
const skyMat = new THREE.MeshBasicNodeMaterial({ fog: false })
skyMat.colorNode = mix(skyBottomColorU, skyTopColorU, uv().y)
const skyMesh = new THREE.Mesh(skyGeo, skyMat)
skyMesh.position.set(0, skyHeight / 2, -60)
scene.add(skyMesh)

// ─── Materials ──────────────────────────────────────────────────────────────
// Standard materials for large/important objects
function makeMat(col, rough = 0.85, metal = 0) {
  return new THREE.MeshStandardMaterial({ color: col, roughness: rough, metalness: metal })
}

// Lambert materials for small/distant objects (cheaper shading)
function makeLambert(col) {
  return new THREE.MeshLambertMaterial({ color: col })
}

const grassMat = makeMat('#5ebf40', 0.9)
const grassDarkMat = makeMat('#3da828', 0.9)
const dirtMat = makeMat('#d48c3a', 0.85)
const pathMat = makeMat('#e8c88a', 0.8)
const waterMat = makeMat('#0088cc', 0.15, 0.1)
const trunkMat = makeLambert('#8b5e3a')
const foliageMats = [makeLambert('#4dc636'), makeLambert('#2db82a'), makeLambert('#7dd44a'), makeLambert('#38c850')]
const buildingMats = [makeMat('#ff9e9e'), makeMat('#9ebfff'), makeMat('#ffcf8a'), makeMat('#bf9eff'), makeMat('#8affc0')]
const roofMats = [makeLambert('#e85050'), makeLambert('#5080e8'), makeLambert('#e8a830'), makeLambert('#9850e8'), makeLambert('#50d878')]
const signMat = makeLambert('#f5e8b0')
const cloudMat = makeLambert('#ffffff')
const accentMats = [makeLambert('#ff5078'), makeLambert('#ffaa40'), makeLambert('#60b8ff'), makeLambert('#b070ff'), makeLambert('#40e890')]
const groundMat = makeMat('#80c860', 0.95)
const oceanMat = new THREE.MeshStandardMaterial({ color: '#0078cc', roughness: 0.12, metalness: 0.1, transparent: true, opacity: 0.9 })

// Additional lambert materials for small details
const doorMat = makeLambert('#6b4226')
const winMat = makeLambert('#fffde0')
const lampPostMat = makeLambert('#555555')
const lampBulbMat = makeLambert('#ffeeaa')
const flagMat = makeLambert('#ff8fa3')
const lilyMat = makeLambert('#5dba5a')
const mushStemMat = makeLambert('#e8ddd0')
const balconyMat = makeLambert('#d5c5f7')
const balloonEnvelopeMat = makeMat('#9b7dff', 0.75)
const balloonStripeMat = makeMat('#efe4ff', 0.82)
const balloonBasketMat = makeMat('#8f5c3a', 0.9)
const balloonRopeMat = makeLambert('#f7f1d7')
const turtleShellMat = makeMat('#4d6b22', 0.82)
const turtleSkinMat = makeMat('#5dc86e', 0.9)
const turtleShellPatternMat = makeMat('#243d0e', 0.85)
const turtleBellyMat = makeLambert('#c2dba0')
const turtleEyeMat = makeLambert('#18180a')
const turtleEyeGleamMat = makeLambert('#f0f0f0')
const seahorseBodyMat = makeMat('#f2a16d', 0.88)
const seahorseFinMat = makeLambert('#ffd4b5')
const dolphinBodyMat = makeMat('#7ca6c8', 0.78)
const dolphinBellyMat = makeLambert('#dbe8f2')

// ─── Shared geometries ──────────────────────────────────────────────────────
const sphereGeo = new THREE.SphereGeometry(1, 24, 18)
const boxGeo = new THREE.BoxGeometry(1, 1, 1)
const cylGeo = new THREE.CylinderGeometry(1, 1, 1, 16)
const coneGeo = new THREE.ConeGeometry(1, 1, 16)
const torusGeo = new THREE.TorusGeometry(1, 0.35, 12, 24)
const signTextTextureCache = new Map()
const posterTexture = new THREE.TextureLoader().load('res/STREAM Poster 2026.JPG')
posterTexture.colorSpace = THREE.SRGBColorSpace

// ─── Geometry merge collector ───────────────────────────────────────────────
// Collects geometries per material for batch merging
const mergeCollector = new Map()

function collectGeo(geo, mat, pos, scale, rot) {
  const g = geo.clone()

  // Apply scale
  if (scale) {
    const sx = typeof scale === 'number' ? scale : scale[0]
    const sy = typeof scale === 'number' ? scale : scale[1]
    const sz = typeof scale === 'number' ? scale : scale[2]
    g.scale(sx, sy, sz)
  }

  // Apply rotation
  if (rot) {
    const m = new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(rot[0], rot[1], rot[2]))
    g.applyMatrix4(m)
  }

  // Apply position
  if (pos) {
    g.translate(pos[0], pos[1], pos[2])
  }

  if (!mergeCollector.has(mat)) mergeCollector.set(mat, [])
  mergeCollector.get(mat).push(g)
}

// Collect geometry with a parent group's transform applied
function collectGeoInGroup(geo, mat, localPos, scale, rot, group) {
  const g = geo.clone()

  // Build local transform matrix
  const localMatrix = new THREE.Matrix4()
  const euler = rot ? new THREE.Euler(rot[0], rot[1], rot[2]) : new THREE.Euler()
  const quat = new THREE.Quaternion().setFromEuler(euler)
  const s = scale
    ? (typeof scale === 'number' ? new THREE.Vector3(scale, scale, scale) : new THREE.Vector3(scale[0], scale[1], scale[2]))
    : new THREE.Vector3(1, 1, 1)
  const p = localPos ? new THREE.Vector3(localPos[0], localPos[1], localPos[2]) : new THREE.Vector3()
  localMatrix.compose(p, quat, s)

  // Apply group world transform
  group.updateWorldMatrix(true, false)
  const worldMatrix = new THREE.Matrix4().multiplyMatrices(group.matrixWorld, localMatrix)
  g.applyMatrix4(worldMatrix)

  if (!mergeCollector.has(mat)) mergeCollector.set(mat, [])
  mergeCollector.get(mat).push(g)
}

// Flush all collected geometries into merged meshes
function flushMergedGeometries() {
  for (const [mat, geos] of mergeCollector) {
    if (geos.length === 0) continue
    const merged = mergeGeometries(geos, false)
    if (!merged) continue
    const mesh = new THREE.Mesh(merged, mat)
    mesh.castShadow = true
    mesh.receiveShadow = true
    scene.add(mesh)
    // Dispose individual geos
    for (const g of geos) g.dispose()
  }
  mergeCollector.clear()
}

// ─── Ocean / Water base ─────────────────────────────────────────────────────
const farOceanGeo = new THREE.PlaneGeometry(300, 300)
const farOcean = new THREE.Mesh(farOceanGeo, oceanMat)
farOcean.rotation.x = -Math.PI / 2
farOcean.position.y = -0.45
farOcean.receiveShadow = true
farOcean.name = 'farOcean'
scene.add(farOcean)

// Interactive water plane
const waterCenter = new THREE.Vector3(0, -0.18, -3)
let waterPlane = null
let deferredInitDone = false
function deferredInit() {
  if (deferredInitDone) return
  deferredInitDone = true
  waterPlane = new WaterPlane(scene, renderer, {
    sizeX: 80, sizeZ: 80, center: waterCenter,
    color: '#0088dd', metalness: 0.05, roughness: 0.08,
    fresnelBias: 0.25, fresnelPower: 1.5, fresnelScale: 1.2,
    resolution: 128, viscosity: 0.6, damping: 0, speed: 0.97,
    mouseDeep: 0.04, mouseSize: 1.2, colliderStrength: 0.002,
    noiseAmplitude: 0.117, noiseFrequency: 4, noiseSpeed: 1.2,
  })
}

// Mouse tracking
const mouseNDC = new THREE.Vector2(9999, 9999)
const mouseNDCIdle = new THREE.Vector2(9999, 9999) // reusable idle vector
let mouseActive = false
window.addEventListener('pointermove', (e) => {
  mouseNDC.x = (e.clientX / innerWidth) * 2 - 1
  mouseNDC.y = -(e.clientY / innerHeight) * 2 + 1
  mouseActive = true
})
window.addEventListener('pointerleave', () => {
  mouseNDC.set(9999, 9999)
  mouseActive = false
})

// ─── Helper: Create an island (kept as live meshes — they're large & few) ───
function createIsland(x, z, radius, height = 1.2, grassColor, scaleX = 1, scaleZ = 1) {
  const group = new THREE.Group();
  group.position.set(x, 0, z);

  // The Top (Grass)
  const topGeo = new THREE.SphereGeometry(radius, 28, 20, 0, Math.PI * 2, 0, Math.PI * 0.55);
  const topMesh = new THREE.Mesh(topGeo, grassColor || grassMat);
  
  // Apply the oval scaling here
  topMesh.scale.set(scaleX, height / radius * 0.7, scaleZ); 
  
  topMesh.position.y = 0;
  topMesh.castShadow = true;
  topMesh.receiveShadow = true;
  group.add(topMesh);

  // The Bottom (Dirt)
  const bottomGeo = new THREE.CylinderGeometry(radius * 0.95, radius * 0.5, height * 1.2, 24);
  const bottomMesh = new THREE.Mesh(bottomGeo, dirtMat);
  
  // Apply the same horizontal scaling to the bottom
  bottomMesh.scale.set(scaleX, 1, scaleZ);
  
  bottomMesh.position.y = -height * 0.6;
  bottomMesh.castShadow = true;
  bottomMesh.receiveShadow = true;
  group.add(bottomMesh);

  scene.add(group);
  return group;
}

// ─── Helper: Stylized tree (merged) ─────────────────────────────────────────
function createTree(parent, x, z, trunkH = 1.2, foliageR = 0.6, foliageType = 'round') {
  collectGeoInGroup(cylGeo, trunkMat, [x, trunkH / 2, z], [0.12, trunkH, 0.12], null, parent)

  const fMat = foliageMats[Math.floor(Math.random() * foliageMats.length)]
  const y = trunkH + foliageR * 0.5

  if (foliageType === 'round') {
    collectGeoInGroup(sphereGeo, fMat, [x, y, z], foliageR, null, parent)
    collectGeoInGroup(sphereGeo, fMat, [x + foliageR * 0.4, y - 0.15, z + 0.1], foliageR * 0.7, null, parent)
    collectGeoInGroup(sphereGeo, fMat, [x - foliageR * 0.3, y + 0.1, z - 0.15], foliageR * 0.55, null, parent)
  } else if (foliageType === 'cone') {
    collectGeoInGroup(coneGeo, fMat, [x, y + 0.2, z], [foliageR * 1.2, foliageR * 2.2, foliageR * 1.2], null, parent)
    collectGeoInGroup(coneGeo, fMat, [x, y + foliageR * 1.4, z], [foliageR * 0.8, foliageR * 1.6, foliageR * 0.8], null, parent)
  }
}

// ─── Helper: Stylized building (merged) ─────────────────────────────────────
function createBuilding(parent, x, z, w, h, d, bIdx = 0) {
  const bMat = buildingMats[bIdx % buildingMats.length]
  const rMat = roofMats[bIdx % roofMats.length]

  collectGeoInGroup(boxGeo, bMat, [x, h / 2, z], [w, h, d], null, parent)
  collectGeoInGroup(coneGeo, rMat, [x, h + 0.35, z], [w * 0.75, 0.7, d * 0.75], null, parent)
  collectGeoInGroup(boxGeo, doorMat, [x, 0.25, z + d / 2 + 0.01], [w * 0.25, 0.5, 0.05], null, parent)

  if (h > 1) {
    collectGeoInGroup(boxGeo, winMat, [x - w * 0.25, h * 0.65, z + d / 2 + 0.01], [0.2, 0.2, 0.03], null, parent)
    collectGeoInGroup(boxGeo, winMat, [x + w * 0.25, h * 0.65, z + d / 2 + 0.01], [0.2, 0.2, 0.03], null, parent)
  }
}

// ─── Helper: Floating sign (merged) ─────────────────────────────────────────
function createSign(parent, x, y, z, text, rotY = 0) {
  // We need a temporary group to compute world positions
  const g = new THREE.Group()
  g.position.set(x, y, z)
  g.rotation.y = rotY
  parent.add(g)
  parent.updateWorldMatrix(true, false)
  g.updateWorldMatrix(true, false)

  collectGeoInGroup(cylGeo, trunkMat, [0, -0.5, 0], [0.05, 1, 0.05], null, g)
  const boardW = text.length * 0.22 + 0.3
  collectGeoInGroup(boxGeo, signMat, [0, 0.15, 0], [boardW, 0.5, 0.06], null, g)
  const textMesh = new THREE.Mesh(
    new THREE.PlaneGeometry(boardW * 0.84, 0.44),
    new THREE.MeshBasicMaterial({
      map: getSignTextTexture(text),
      transparent: true,
      alphaTest: 0.01,
      toneMapped: false, // ADD THIS LINE
      side: THREE.DoubleSide
    })
  )
  textMesh.position.set(0, 0.15, 0.035)
  g.add(textMesh)
}

function createPoster(parent, x, y, z, texture, options = {}) {
  const {
    rotY = 0,
    width = 2.4,
    aspect = 3756 / 1518,
  } = options

  const height = width / aspect
  const g = new THREE.Group()
  g.position.set(x, y, z)
  g.rotation.y = rotY
  parent.add(g)
  parent.updateWorldMatrix(true, false)
  g.updateWorldMatrix(true, false)

  collectGeoInGroup(cylGeo, trunkMat, [-width * 0.32, -height * 0.35, -0.08], [0.06, height * 1.35, 0.06], null, g)
  collectGeoInGroup(cylGeo, trunkMat, [width * 0.32, -height * 0.35, -0.08], [0.06, height * 1.35, 0.06], null, g)
  collectGeoInGroup(boxGeo, signMat, [0, 0, -0.02], [width + 0.16, height + 0.16, 0.06], null, g)

  const posterMesh = new THREE.Mesh(
    new THREE.PlaneGeometry(width, height),
    new THREE.MeshBasicMaterial({
      map: texture,
      toneMapped: false,
      side: THREE.DoubleSide,
    })
  )
  posterMesh.position.set(0, 0, 0.02)
  g.add(posterMesh)
}

// ─── Helper: Bridge/Path (merged) ───────────────────────────────────────────
function createBridge(x1, z1, x2, z2, y = 0.15) {
  const dx = x2 - x1, dz = z2 - z1
  const len = Math.sqrt(dx * dx + dz * dz)
  const angle = Math.atan2(dx, dz)
  const cx = (x1 + x2) / 2, cz = (z1 + z2) / 2

  const pathGeo = new THREE.BoxGeometry(0.6, 0.12, len)
  collectGeo(pathGeo, pathMat, [cx, y, cz], null, [0, angle, 0])

  // Railing posts — small, no shadows needed
  const steps = Math.floor(len / 1.2)
  for (let i = 0; i <= steps; i++) {
    const t = i / steps
    const px = x1 + dx * t
    const pz = z1 + dz * t
    for (const side of [-1, 1]) {
      const ox = Math.cos(angle) * 0.35 * side
      const oz = -Math.sin(angle) * 0.35 * side
      collectGeo(cylGeo, trunkMat, [px + ox, y + 0.2, pz + oz], [0.03, 0.4, 0.03], null)
    }
  }
}

// ─── Bouncing objects tracker ───────────────────────────────────────────────
const bouncingObjects = []
function markBouncing(mesh, baseY, amplitude = 0.15, speed = 1.5, phase = 0) {
  bouncingObjects.push({ mesh, baseY, amplitude, speed, phase })
}

// ─── Helper: Add live mesh (for animated/bouncing objects only) ─────────────
function addLiveMesh(geo, mat, pos, scale, rot, parent = scene) {
  const m = new THREE.Mesh(geo, mat)
  if (pos) m.position.set(...pos)
  if (scale) {
    if (typeof scale === 'number') m.scale.setScalar(scale)
    else m.scale.set(...scale)
  }
  if (rot) m.rotation.set(...rot)
  m.castShadow = false
  m.receiveShadow = false
  parent.add(m)
  return m
}

function getSignTextTexture(text) {
  if (signTextTextureCache.has(text)) return signTextTextureCache.get(text)

  const canvas = document.createElement('canvas')
  canvas.width = 612
  canvas.height = 150
  const ctx = canvas.getContext('2d')

  ctx.clearRect(0, 0, canvas.width, canvas.height)
  ctx.font = '700 90px Quicksand, "Segoe UI Emoji", "Apple Color Emoji", sans-serif'
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.lineJoin = 'round'
  ctx.strokeStyle = '#000000' //'#fff7da'
  ctx.lineWidth = 14
  ctx.strokeText(text, canvas.width / 2, canvas.height / 2)
  ctx.fillStyle = '#ffffff' //'#3da828' // '#fff7da' //'#6a4325'
  ctx.fillText(text, canvas.width / 2, canvas.height / 2)
  //ctx.fillText(text, canvas.width / 2, canvas.height / 2); // Draw twice for better opacity

  const texture = new THREE.CanvasTexture(canvas)
  // ADD THIS LINE:
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.needsUpdate = true
  signTextTextureCache.set(text, texture)
  return texture
}

// ─── Helper: Abstract decoration ────────────────────────────────────────────
// Static ones go to merge collector, animated ones stay live
function createAbstractObjectMerged(parent, x, y, z, type = 'torus') {
  const aMat = accentMats[Math.floor(Math.random() * accentMats.length)]
  if (type === 'torus') {
    collectGeoInGroup(torusGeo, aMat, [x, y, z], 0.3, [Math.PI / 2, 0, 0], parent)
  } else if (type === 'diamond') {
    collectGeoInGroup(boxGeo, aMat, [x, y, z], 0.35, [Math.PI / 4, Math.PI / 4, 0], parent)
  } else if (type === 'sphere') {
    collectGeoInGroup(sphereGeo, aMat, [x, y, z], 0.25, null, parent)
  }
}

function createAbstractObjectLive(parent, x, y, z, type = 'torus') {
  const aMat = accentMats[Math.floor(Math.random() * accentMats.length)]
  if (type === 'torus') {
    return addLiveMesh(torusGeo, aMat, [x, y, z], 0.3, [Math.PI / 2, 0, 0], parent)
  } else if (type === 'diamond') {
    return addLiveMesh(boxGeo, aMat, [x, y, z], 0.35, [Math.PI / 4, Math.PI / 4, 0], parent)
  } else if (type === 'sphere') {
    return addLiveMesh(sphereGeo, aMat, [x, y, z], 0.25, null, parent)
  }
}

function createBalloon(config = { body: 0xe74c3c }, position = [0, 5, 11]) {
  const balloonGroup = new THREE.Group();
  
  // 1. Envelope (The big balloon)
  const balloonMat = new THREE.MeshStandardMaterial({ 
    color: config.body, 
    flatShading: true, 
    roughness: 0.7 
  });
  const envelope = new THREE.Mesh(sphereGeo, balloonMat);
  // Reduced from [1.5, 2.0, 1.5] to [0.7, 0.9, 0.7]
  envelope.scale.set(0.7, 0.9, 0.7); 
  envelope.position.set(0, 0.8, 0); 
  envelope.castShadow = true;
  balloonGroup.add(envelope);

  // 2. Accent Stripes (Torus)
  for (let i = -1; i <= 1; i++) {
    // Reduced radius from 1.4 to 0.65
    const bandGeom = new THREE.TorusGeometry(0.65, 0.02, 6, 24);
    const band = new THREE.Mesh(bandGeom, dolphinBellyMat);
    band.position.set(0, 0.8 + i * 0.25, 0);
    band.rotation.x = Math.PI / 2;
    balloonGroup.add(band);
  }

  // 3. Ropes
  const ropeMat = new THREE.MeshStandardMaterial({ color: 0x8B7355 });
  const ropePositions = [[0.2, 0.2], [-0.2, 0.2], [0.2, -0.2], [-0.2, -0.2]];
  for (const rp of ropePositions) {
    const rope = new THREE.Mesh(cylGeo, ropeMat);
    rope.scale.set(0.01, 0.6, 0.01);
    rope.position.set(rp[0], 0.2, rp[1]);
    balloonGroup.add(rope);
  }

  // 4. Basket
  const basketMat = new THREE.MeshStandardMaterial({ color: 0x8B6914 });
  const basket = new THREE.Mesh(boxGeo, basketMat);
  // Reduced scale from 0.8 to 0.35
  basket.scale.set(0.35, 0.25, 0.35);
  basket.position.set(0, -0.15, 0);
  basket.castShadow = true;
  balloonGroup.add(basket);

  // 5. Flame
  const flameGeom = new THREE.ConeGeometry(0.08, 0.2, 6);
  const flameMat = new THREE.MeshStandardMaterial({ 
    color: 0xff6600, 
    emissive: 0xff4400, 
    emissiveIntensity: 1.0, 
    transparent: true, 
    opacity: 0.8 
  });
  const flame = new THREE.Mesh(flameGeom, flameMat);
  flame.position.set(0, 0, 0);
  balloonGroup.add(flame);

  balloonGroup.position.set(...position);
  scene.add(balloonGroup);

  return { mesh: balloonGroup, flame };
}

function createHotAirBalloon(options = {}) {
  const {
    envelopeScale = [1.45, 1.8, 1.45],
    stripeScale = [0.35, 2.2, 3.0],
    basketScale = [0.8, 0.6, 0.8],
    position = [0, 5.6, 11],
    scale = 0.1625,
    envelopeMat = balloonEnvelopeMat,
    stripeMat = balloonStripeMat,
  } = options

  const balloon = new THREE.Group()

  // 1. Improved Envelope Shape (The "Upper Part")
  // We use the shared sphereGeo but lift it so it sits above the center
  const envelope = new THREE.Mesh(sphereGeo, envelopeMat)
  envelope.scale.set(1.45, 1.9, 1.45) // Pixar-style vertical stretch
  envelope.position.set(0, 0, 0)    // Lifted up
  envelope.castShadow = true
  envelope.receiveShadow = true
  balloon.add(envelope)

  // 2. Horizontal Accent Stripes (Replacing the box stripes)
  // These use the Torus geometry for a much smoother, rounded look
  for (let i = -1; i <= 1; i++) {
    const bandGeom = new THREE.TorusGeometry(1.42, 0.06, 6, 24)
    const band = new THREE.Mesh(bandGeom, stripeMat)
    band.position.set(0, i * 0.7, 0)
    band.rotation.x = Math.PI / 2
    band.scale.set(1, 0.95, 1) // Slightly flatten the rings
    balloon.add(band)
  }

  const basket = new THREE.Mesh(boxGeo, balloonBasketMat)
  basket.position.set(0, -2.5, 0)
  basket.scale.set(...basketScale)
  basket.castShadow = true
  basket.receiveShadow = true
  balloon.add(basket)

  const ropeOffsets = [
    [-0.42, -1.35, -0.42],
    [0.42, -1.35, -0.42],
    [-0.42, -1.35, 0.42],
    [0.42, -1.35, 0.42],
  ]

  for (const [x, y, z] of ropeOffsets) {
    const rope = new THREE.Mesh(cylGeo, balloonRopeMat)
    rope.position.set(x, y - 0.45, z)
    rope.scale.set(0.02, 1.1, 0.02)
    rope.castShadow = false
    rope.receiveShadow = false
    balloon.add(rope)
  }

  balloon.position.set(...position)
  balloon.scale.setScalar(scale)
  scene.add(balloon)
  return balloon
}

function createTurtle(position = [8.0, -0.15, 8.5], scale = 0.22) {
  const turtle = new THREE.Group()

  const shell = new THREE.Mesh(sphereGeo, turtleShellMat)
  shell.scale.set(1.08, 0.42, 1.36)
  shell.castShadow = true
  shell.receiveShadow = true
  turtle.add(shell)

  const shellCap = new THREE.Mesh(sphereGeo, turtleShellPatternMat)
  shellCap.position.y = 0.12
  shellCap.scale.set(0.72, 0.18, 0.96)
  shellCap.castShadow = true
  turtle.add(shellCap)

  const shellRidge = new THREE.Mesh(boxGeo, turtleShellPatternMat)
  shellRidge.position.set(0, 0.1, 0)
  shellRidge.scale.set(0.14, 0.1, 1.2)
  shellRidge.castShadow = true
  turtle.add(shellRidge)

  const costalPositions = [
    [-0.48, 0.24, 0.46], [0.48, 0.24, 0.46],
    [-0.54, 0.22, 0.04], [0.54, 0.22, 0.04],
    [-0.48, 0.20, -0.42], [0.48, 0.20, -0.42],
  ]
  for (const [x, y, z] of costalPositions) {
    const scute = new THREE.Mesh(sphereGeo, turtleShellPatternMat)
    scute.position.set(x, y, z)
    scute.scale.set(0.28, 0.07, 0.3)
    turtle.add(scute)
  }

  const plastron = new THREE.Mesh(sphereGeo, turtleBellyMat)
  plastron.position.y = -0.14
  plastron.scale.set(0.92, 0.1, 1.18)
  turtle.add(plastron)

  const neck = new THREE.Mesh(cylGeo, turtleSkinMat)
  neck.position.set(0, 0, 0.92)
  neck.rotation.x = Math.PI / 2
  neck.scale.set(0.12, 0.28, 0.12)
  neck.castShadow = true
  turtle.add(neck)

  const head = new THREE.Mesh(sphereGeo, turtleSkinMat)
  head.position.set(0, 0.02, 1.26)
  head.scale.set(0.34, 0.2, 0.44)
  head.castShadow = true
  turtle.add(head)

  const snout = new THREE.Mesh(sphereGeo, turtleSkinMat)
  snout.position.set(0, 0.01, 1.65)
  snout.scale.set(0.1, 0.08, 0.1)
  turtle.add(snout)

  for (const side of [-1, 1]) {
    const eye = new THREE.Mesh(sphereGeo, turtleEyeMat)
    eye.position.set(side * 0.3, 0.1, 1.3)
    eye.scale.setScalar(0.065)
    turtle.add(eye)
    const gleam = new THREE.Mesh(sphereGeo, turtleEyeGleamMat)
    gleam.position.set(side * 0.32, 0.12, 1.36)
    gleam.scale.setScalar(0.022)
    turtle.add(gleam)
  }

  const tail = new THREE.Mesh(coneGeo, turtleSkinMat)
  tail.position.set(0, -0.04, -1.18)
  tail.rotation.x = Math.PI / 2
  tail.scale.set(0.08, 0.18, 0.08)
  tail.castShadow = true
  turtle.add(tail)

  const legOffsets = [
    [-0.82, -0.12, 0.58],
    [0.82, -0.12, 0.58],
    [-0.74, -0.14, -0.52],
    [0.74, -0.14, -0.52],
  ]
  const legs = []
  for (const [index, [x, y, z]] of legOffsets.entries()) {
    const leg = new THREE.Mesh(sphereGeo, turtleSkinMat)
    leg.position.set(x, y, z)
    const isFront = index < 2
    leg.scale.set(isFront ? 0.34 : 0.26, 0.05, isFront ? 0.48 : 0.34)
    leg.rotation.x = isFront ? -0.2 : 0.1
    leg.castShadow = true
    turtle.add(leg)
    legs.push(leg)
  }

  turtle.position.set(...position)
  turtle.scale.setScalar(scale)
  scene.add(turtle)

  return { mesh: turtle, head, neck, legs }
}

function createSeahorse(position = [2.4, -0.18, 12.4], scale = 0.198) {
  const seahorse = new THREE.Group()

  const torso = new THREE.Mesh(sphereGeo, seahorseBodyMat)
  torso.scale.set(0.42, 0.78, 0.28)
  torso.castShadow = true
  seahorse.add(torso)

  const belly = new THREE.Mesh(sphereGeo, seahorseFinMat)
  belly.position.set(0.08, -0.02, 0.12)
  belly.scale.set(0.18, 0.58, 0.16)
  belly.castShadow = true
  seahorse.add(belly)

  const neck = new THREE.Mesh(cylGeo, seahorseBodyMat)
  neck.position.set(0.02, 0.56, 0.02)
  neck.rotation.z = -0.28
  neck.scale.set(0.12, 0.42, 0.12)
  neck.castShadow = true
  seahorse.add(neck)

  const head = new THREE.Mesh(sphereGeo, seahorseBodyMat)
  head.position.set(0.16, 0.92, 0.04)
  head.scale.set(0.22, 0.2, 0.24)
  head.castShadow = true
  seahorse.add(head)

  const snout = new THREE.Mesh(cylGeo, seahorseBodyMat)
  snout.position.set(0.34, 0.88, 0.05)
  snout.rotation.z = Math.PI / 2
  snout.scale.set(0.06, 0.24, 0.06)
  snout.castShadow = true
  seahorse.add(snout)

  const crest = new THREE.Mesh(coneGeo, seahorseFinMat)
  crest.position.set(0.05, 1.08, 0.02)
  crest.rotation.z = -0.15
  crest.scale.set(0.08, 0.2, 0.08)
  crest.castShadow = true
  seahorse.add(crest)

  const fin = new THREE.Mesh(boxGeo, seahorseFinMat)
  fin.position.set(-0.12, 0.28, 0)
  fin.rotation.z = 0.3
  fin.scale.set(0.05, 0.34, 0.22)
  fin.castShadow = true
  seahorse.add(fin)

  const tail = new THREE.Group()
  tail.position.set(-0.04, -0.7, 0)
  const tailSegments = []
  for (let i = 0; i < 4; i++) {
    const segment = new THREE.Mesh(sphereGeo, seahorseBodyMat)
    segment.position.set(-0.08 * i, -0.18 * i, 0)
    segment.scale.set(0.16 - i * 0.02, 0.13 - i * 0.015, 0.16 - i * 0.02)
    segment.castShadow = true
    tail.add(segment)
    tailSegments.push(segment)
  }
  seahorse.add(tail)

  seahorse.position.set(...position)
  seahorse.scale.setScalar(scale)
  seahorse.rotation.y = -0.5
  scene.add(seahorse)

  return { mesh: seahorse, head, fin, tail, tailSegments }
}

function createDolphin(position = [0.9, -0.08, 12.9], scale = 0.42) {
  const dolphin = new THREE.Group()

  const body = new THREE.Mesh(sphereGeo, dolphinBodyMat)
  body.scale.set(1.45, 0.34, 0.34)
  body.castShadow = true
  dolphin.add(body)

  const shoulder = new THREE.Mesh(sphereGeo, dolphinBodyMat)
  shoulder.position.set(0.35, 0.04, 0)
  shoulder.scale.set(0.82, 0.24, 0.24)
  shoulder.castShadow = true
  dolphin.add(shoulder)

  const melon = new THREE.Mesh(sphereGeo, dolphinBodyMat)
  melon.position.set(0.98, 0.1, 0)
  melon.scale.set(0.42, 0.24, 0.24)
  melon.castShadow = true
  dolphin.add(melon)

  const belly = new THREE.Mesh(sphereGeo, dolphinBellyMat)
  belly.position.set(0.25, -0.1, 0)
  belly.scale.set(1.08, 0.16, 0.2)
  belly.castShadow = true
  dolphin.add(belly)

  const snout = new THREE.Mesh(cylGeo, dolphinBodyMat)
  snout.position.set(1.45, 0.02, 0)
  snout.rotation.z = Math.PI / 2
  snout.scale.set(0.07, 0.5, 0.07)
  snout.castShadow = true
  dolphin.add(snout)

  const lowerJaw = new THREE.Mesh(cylGeo, dolphinBellyMat)
  lowerJaw.position.set(1.43, -0.05, 0)
  lowerJaw.rotation.z = Math.PI / 2
  lowerJaw.scale.set(0.045, 0.42, 0.05)
  lowerJaw.castShadow = true
  dolphin.add(lowerJaw)

  const dorsalFin = new THREE.Mesh(coneGeo, dolphinBodyMat)
  dorsalFin.position.set(-0.18, 0.32, 0)
  dorsalFin.rotation.z = -0.2
  dorsalFin.scale.set(0.1, 0.26, 0.08)
  dorsalFin.castShadow = true
  dolphin.add(dorsalFin)

  const sideFinLeft = new THREE.Mesh(coneGeo, dolphinBodyMat)
  sideFinLeft.position.set(0.08, -0.06, 0.26)
  sideFinLeft.rotation.x = Math.PI / 2
  sideFinLeft.rotation.z = -0.95
  sideFinLeft.scale.set(0.08, 0.34, 0.12)
  sideFinLeft.castShadow = true
  dolphin.add(sideFinLeft)

  const sideFinRight = sideFinLeft.clone()
  sideFinRight.position.z = -0.28
  sideFinRight.rotation.z = 0.95
  dolphin.add(sideFinRight)

  const tailPeduncle = new THREE.Mesh(cylGeo, dolphinBodyMat)
  tailPeduncle.position.set(-1.45, -0.02, 0)
  tailPeduncle.rotation.z = Math.PI / 2
  tailPeduncle.scale.set(0.07, 0.52, 0.07)
  tailPeduncle.castShadow = true
  dolphin.add(tailPeduncle)

  const tail = new THREE.Group()
  tail.position.set(-1.9, 0, 0)

  const flukeTop = new THREE.Mesh(coneGeo, dolphinBodyMat)
  flukeTop.rotation.z = -Math.PI / 2.35
  flukeTop.position.set(-0.03, 0.18, 0)
  flukeTop.scale.set(0.08, 0.42, 0.11)
  flukeTop.castShadow = true
  tail.add(flukeTop)

  const flukeBottom = flukeTop.clone()
  flukeBottom.position.y = -0.18
  flukeBottom.rotation.z = Math.PI / 2.35
  tail.add(flukeBottom)
  dolphin.add(tail)

  dolphin.position.set(...position)
  dolphin.scale.setScalar(scale)
  dolphin.rotation.y = 0.4
  scene.add(dolphin)

  return { mesh: dolphin, tail, dorsalFin, sideFins: [sideFinLeft, sideFinRight] }
}

function createDolphin2(position = [0.9, -0.08, 12.9], scale = 0.42) {
  const dolphin = new THREE.Group()

  // Main Body - Streamlined and slightly thicker for a "friendly" look
  const body = new THREE.Mesh(sphereGeo, dolphinBodyMat)
  body.scale.set(1.6, 0.45, 0.4) 
  body.castShadow = true
  dolphin.add(body)

  // Melon (Forehead) - Integrated more smoothly
  const melon = new THREE.Mesh(sphereGeo, dolphinBodyMat)
  melon.position.set(0.65, 0.15, 0)
  melon.scale.set(0.45, 0.35, 0.32)
  dolphin.add(melon)

  // Belly - Uses the lighter material for contrast
  const belly = new THREE.Mesh(sphereGeo, dolphinBellyMat)
  belly.position.set(0.1, -0.15, 0)
  belly.scale.set(1.2, 0.25, 0.3)
  dolphin.add(belly)

  // Snout - Slightly shorter and rounded
  const snout = new THREE.Mesh(cylGeo, dolphinBodyMat)
  snout.position.set(1.1, 0.05, 0)
  snout.rotation.z = Math.PI / 2
  snout.scale.set(0.08, 0.4, 0.08)
  dolphin.add(snout)

  // Improved Dorsal Fin - Swept back
  const dorsalFin = new THREE.Mesh(coneGeo, dolphinBodyMat)
  dorsalFin.position.set(-0.2, 0.4, 0)
  dorsalFin.rotation.z = -0.6 
  dorsalFin.scale.set(0.12, 0.35, 0.08)
  dolphin.add(dorsalFin)

  // Side Fins (Pectoral Fins)
  const sideFinLeft = new THREE.Mesh(coneGeo, dolphinBodyMat)
  sideFinLeft.position.set(0.2, -0.1, 0.3)
  sideFinLeft.rotation.set(1.2, 0, -0.5)
  sideFinLeft.scale.set(0.1, 0.4, 0.15)
  dolphin.add(sideFinLeft)

  const sideFinRight = sideFinLeft.clone()
  sideFinRight.position.z = -0.3
  sideFinRight.rotation.set(-1.2, 0, -0.5)
  dolphin.add(sideFinRight)

  // Tail Group with Horizontal Flukes (Dolphins move tails vertically)
  const tail = new THREE.Group()
  tail.position.set(-1.4, 0, 0)
  
  const flukes = new THREE.Mesh(boxGeo, dolphinBodyMat)
  flukes.scale.set(0.4, 0.05, 1.1) 
  tail.add(flukes)
  dolphin.add(tail)

  dolphin.position.set(...position)
  dolphin.scale.setScalar(scale)
  scene.add(dolphin)

  return { mesh: dolphin, tail, dorsalFin, sideFins: [sideFinLeft, sideFinRight] }
}

function createDolphin1(position = [0.9, -0.08, 12.9], scale = 0.42) {
  const dolphin = new THREE.Group();

  // --- BODY SEGMENTS (The "Spine" approach) ---
  
  // 1. CHEST (The thickest part)
  const chest = new THREE.Mesh(sphereGeo, dolphinBodyMat);
  chest.scale.set(0.8, 0.55, 0.45);
  chest.position.set(0.3, 0, 0);
  chest.castShadow = true;
  dolphin.add(chest);

  // 2. MID-BODY (Slightly smaller, blending backward)
  const midBody = new THREE.Mesh(sphereGeo, dolphinBodyMat);
  midBody.scale.set(0.9, 0.45, 0.38);
  midBody.position.set(-0.3, -0.02, 0);
  dolphin.add(midBody);

  // 3. TAIL STOCK (The thin part leading to the flukes)
  const tailStock = new THREE.Mesh(sphereGeo, dolphinBodyMat);
  tailStock.scale.set(0.8, 0.25, 0.18);
  tailStock.position.set(-1.0, -0.05, 0);
  dolphin.add(tailStock);

  // --- HEAD & SNOUT ---

  // MELON (The forehead - slightly more bulbous and forward)
  const melon = new THREE.Mesh(sphereGeo, dolphinBodyMat);
  melon.position.set(0.75, 0.18, 0);
  melon.scale.set(0.4, 0.35, 0.32);
  dolphin.add(melon);

  // BEAK (The snout - thinner and integrated lower)
  const beak = new THREE.Mesh(sphereGeo, dolphinBodyMat);
  beak.position.set(1.15, 0.02, 0);
  beak.scale.set(0.45, 0.12, 0.12);
  dolphin.add(beak);

  // BELLY (White patch - scaled to fit the new chest/mid sections)
  const belly = new THREE.Mesh(sphereGeo, dolphinBellyMat);
  belly.position.set(0.2, -0.18, 0);
  belly.scale.set(1.4, 0.25, 0.35);
  dolphin.add(belly);

  // --- FINS (Refined angles) ---

  // DORSAL FIN (Moved further back, more "hooked")
  const dorsalFin = new THREE.Mesh(coneGeo, dolphinBodyMat);
  dorsalFin.position.set(-0.25, 0.42, 0);
  dorsalFin.rotation.z = -0.9;
  dorsalFin.scale.set(0.06, 0.5, 0.04); 
  dolphin.add(dorsalFin);

  // SIDE FINS (Longer and more swept back)
  const sideFinLeft = new THREE.Mesh(coneGeo, dolphinBodyMat);
  sideFinLeft.position.set(0.45, -0.15, 0.35);
  sideFinLeft.rotation.set(1.1, 0, -0.8);
  sideFinLeft.scale.set(0.07, 0.45, 0.15);
  dolphin.add(sideFinLeft);

  const sideFinRight = sideFinLeft.clone();
  sideFinRight.position.z = -0.35;
  sideFinRight.rotation.set(-1.1, 0, -0.8);
  dolphin.add(sideFinRight);

  // --- TAIL FLUKES ---
  const tail = new THREE.Group();
  tail.position.set(-1.6, -0.05, 0);

  const flukeL = new THREE.Mesh(coneGeo, dolphinBodyMat);
  flukeL.position.set(0, 0, 0.22);
  flukeL.rotation.set(Math.PI / 2, 0, Math.PI / 1.8);
  flukeL.scale.set(0.12, 0.55, 0.02);
  tail.add(flukeL);

  const flukeR = flukeL.clone();
  flukeR.position.z = -0.22;
  flukeR.rotation.x = -Math.PI / 2;
  tail.add(flukeR);

  dolphin.add(tail);

  // --- EYES ---
  const eyeGeo = new THREE.SphereGeometry(0.025, 12, 12);
  const eyeMat = new THREE.MeshBasicMaterial({ color: 0x000000 });
  const eyeL = new THREE.Mesh(eyeGeo, eyeMat);
  eyeL.position.set(0.85, 0.08, 0.2);
  const eyeR = eyeL.clone();
  eyeR.position.z = -0.2;
  dolphin.add(eyeL, eyeR);

  dolphin.position.set(...position);
  dolphin.scale.setScalar(scale);
  scene.add(dolphin);

  return { mesh: dolphin, tail, dorsalFin, sideFins: [sideFinLeft, sideFinRight] };
}

// ═══════════════════════════════════════════════════════════════════════════
// ─── BUILD THE WORLD ────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

// ─── Island 1: Welcome Island (center-front) ────────────────────────────────
const island1 = createIsland(0, 8, 4, 1.5, grassMat, 1.5, 0.8)
island1.rotation.y = -Math.PI / 16
island1.updateWorldMatrix(true, false)
createTree(island1, -1.5, -0.8, 1.0, 0.55, 'round')
createTree(island1, 1.8, 0.5, 1.3, 0.65, 'round')
createTree(island1, -0.5, 1.5, 0.8, 0.4, 'cone')
createPoster(island1, 0.1, 3.4, -0.2, posterTexture, { rotY: 0.08, width: 10.34 })
createAbstractObjectMerged(island1, 0.8, 1.5, -0.5, 'torus')
markBouncing(createAbstractObjectLive(island1, -1, 1.8, 0.8, 'sphere'), 1.8, 0.2, 1.2, 0)

// Flower patches on island 1 (merged, no shadows)
for (let i = 0; i < 8; i++) {
  const a = Math.random() * Math.PI * 2
  const r = 1 + Math.random() * 2
  const fx = Math.cos(a) * r
  const fz = Math.sin(a) * r
  collectGeoInGroup(sphereGeo, accentMats[i % accentMats.length], [fx, 0.6, fz], 0.08, null, island1)
}

// ─── Island 2: Village Island (left-mid) ────────────────────────────────────
const island2 = createIsland(-8, -2, 4.5, 1.6, grassDarkMat)
island2.updateWorldMatrix(true, false)
createBuilding(island2, -0.8, -0.3, 1.0, 1.4, 0.9, 0)
createBuilding(island2, 1.2, 0.5, 0.8, 1.8, 0.8, 1)
createBuilding(island2, -0.2, 1.5, 0.7, 1.1, 0.7, 2)
createTree(island2, -2.2, 1, 1.1, 0.5, 'cone')
createTree(island2, 2.3, -0.8, 0.9, 0.45, 'round')
createSign(island2, 0, 1.6, 2.5, '🏠 Village', 0.2)

// Lamp post (merged)
collectGeoInGroup(cylGeo, lampPostMat, [1.8, 0.7, 1.5], [0.04, 1.4, 0.04], null, island2)
// Lamp bulb (bouncing — live)
const lamp = addLiveMesh(sphereGeo, lampBulbMat, [1.8, 1.5, 1.5], 0.12, null, island2)
markBouncing(lamp, 1.5, 0.05, 0.8, 1)

// ─── Island 3: Garden Island (right-mid) ────────────────────────────────────
const island3 = createIsland(9, -1, 3.8, 1.4)
island3.updateWorldMatrix(true, false)
createTree(island3, -1, -0.5, 1.5, 0.7, 'round')
createTree(island3, 0.8, 0.8, 1.8, 0.8, 'round')
createTree(island3, -0.3, 1.5, 1.0, 0.5, 'cone')
createTree(island3, 1.5, -1, 0.7, 0.35, 'cone')
createAbstractObjectMerged(island3, 0, 1.3, 0, 'diamond')
markBouncing(createAbstractObjectLive(island3, -1.5, 1.0, 1, 'torus'), 1.0, 0.18, 1.6, 2)
createSign(island3, 0.5, 1.5, 2, 'Garden', -0.15)

// Mushrooms (merged, no shadows)
for (let i = 0; i < 4; i++) {
  const a = Math.random() * Math.PI * 2
  const r = 0.8 + Math.random() * 1.5
  const mx = Math.cos(a) * r, mz = Math.sin(a) * r
  collectGeoInGroup(cylGeo, mushStemMat, [mx, 0.35, mz], [0.06, 0.3, 0.06], null, island3)
  collectGeoInGroup(sphereGeo, accentMats[(i + 1) % accentMats.length], [mx, 0.55, mz], [0.15, 0.1, 0.15], null, island3)
}

// ─── Island 4: Lookout Island (center-back) ─────────────────────────────────
const island4 = createIsland(0, -12, 3.5, 1.8, grassDarkMat)
island4.updateWorldMatrix(true, false)

// Tower (merged)
collectGeoInGroup(cylGeo, buildingMats[3], [0, 1.5, 0], [0.8, 3, 0.8], null, island4)
collectGeoInGroup(coneGeo, roofMats[3], [0, 3.3, 0], [1.1, 0.9, 1.1], null, island4)

// Balcony ring (merged)
collectGeoInGroup(torusGeo, balconyMat, [0, 2.5, 0], [0.55, 0.55, 0.55], [Math.PI / 2, 0, 0], island4)

// Flag pole (merged) + flag (bouncing — live)
collectGeoInGroup(cylGeo, trunkMat, [0, 3.9, 0], [0.03, 0.8, 0.03], null, island4)
const flag = addLiveMesh(boxGeo, flagMat, [0.2, 4.2, 0], [0.35, 0.22, 0.02], null, island4)
markBouncing(flag, 4.2, 0.05, 2.0, 0.5)

createTree(island4, -1.8, 0.5, 1.2, 0.55, 'cone')
createTree(island4, 1.5, -0.8, 1.0, 0.5, 'round')
createSign(island4, 1.2, 1.8, 1.5, 'Lookout', 0.3)

// ─── Island 5: Pond Island (far left-back) ──────────────────────────────────
const island5 = createIsland(-9, -13, 3.2, 1.3)
island5.updateWorldMatrix(true, false)

// Small pond
collectGeoInGroup(cylGeo, waterMat, [0, 0.52, 0], [1.4, 0.1, 1.4], null, island5)

createTree(island5, -1.5, -1, 0.9, 0.4, 'round')
createTree(island5, 1.3, 1.2, 1.1, 0.5, 'cone')
createAbstractObjectMerged(island5, 0.8, 0.9, -0.8, 'sphere')
markBouncing(createAbstractObjectLive(island5, -0.5, 1.2, 1, 'diamond'), 1.2, 0.15, 1.8, 3)
createSign(island5, -0.3, 1.3, 1.8, 'Pond', 0.1)

// Lily pads (merged)
for (let i = 0; i < 3; i++) {
  const a = (i / 3) * Math.PI * 2 + 0.3
  collectGeoInGroup(cylGeo, lilyMat, [Math.cos(a) * 0.6, 0.58, Math.sin(a) * 0.6], [0.2, 0.02, 0.2], null, island5)
}

// ─── Island 6: Far right back ───────────────────────────────────────────────
const island6 = createIsland(10, -13, 3, 1.2, grassDarkMat)
island6.updateWorldMatrix(true, false)
createBuilding(island6, 0, 0, 1.2, 2.0, 1.0, 3)
createTree(island6, -1.5, 0.8, 1.0, 0.5, 'cone')
createTree(island6, 1.3, -0.5, 0.8, 0.4, 'round')
markBouncing(createAbstractObjectLive(island6, -0.5, 1.6, -1, 'torus'), 1.6, 0.12, 1.4, 4)
createSign(island6, 0.8, 1.8, 1.5, 'Studio', -0.2)

// ─── Bridges connecting islands ─────────────────────────────────────────────
createBridge(0, 5, -5.5, 0)
createBridge(0, 5, 6, 1)
createBridge(-5.5, -3, -6.5, -10)
createBridge(6, -2.5, 3, -10)
createBridge(0, -10, -6.5, -11.5)
createBridge(3, -11, 7.5, -12)
createBridge(6.5, -2, 7.5, -11)

// ─── Flush all merged static geometry ───────────────────────────────────────
flushMergedGeometries()

const hotAirBalloons = [
  {
    mesh: createHotAirBalloon({ position: [0, 5.6, 11], scale: 0.1625 }),
    centerX: 0, radiusX: 3.5, radiusZ: 1.8, baseY: 5.6, speed: 0.12, bobSpeed: 0.9, bobAmp: 0.45, centerZ: 11, phase: 0,
  },
]

//const balloon = createBalloon({ body: 0xe74c3c }, [0, 3.6, 11]);

const turtle = createTurtle()
const seahorse = createSeahorse()
//const dolphin = createDolphin()

// ─── Clouds (instanced) ────────────────────────────────────────────────────
const cloudDefs = [
  { x: -12, y: 10, z: -5, s: 1.2 },
  { x: 8, y: 11, z: 3, s: 0.9 },
  { x: -3, y: 12, z: -18, s: 1.5 },
  { x: 14, y: 9.5, z: -10, s: 0.8 },
  { x: -8, y: 13, z: 8, s: 1.0 },
  { x: 5, y: 10.5, z: -25, s: 1.1 },
  { x: -15, y: 11.5, z: -20, s: 0.7 },
]

// Each cloud has 4 sub-spheres; define offsets + scales relative to cloud center
const cloudSubParts = [
  { ox: 0, oy: 0, oz: 0, sx: 1.2, sy: 0.5, sz: 0.8 },
  { ox: 0.7, oy: 0.1, oz: 0.1, sx: 0.8, sy: 0.4, sz: 0.6 },
  { ox: -0.6, oy: 0.05, oz: -0.1, sx: 0.9, sy: 0.45, sz: 0.7 },
  { ox: 0.3, oy: 0.15, oz: -0.3, sx: 0.6, sy: 0.35, sz: 0.5 },
]

const totalCloudInstances = cloudDefs.length * cloudSubParts.length
const cloudInstancedMesh = new THREE.InstancedMesh(sphereGeo, cloudMat, totalCloudInstances)
cloudInstancedMesh.castShadow = false
cloudInstancedMesh.receiveShadow = false

const cloudInstanceData = [] // per cloud: { baseX, baseY, speed, drift, indices[] }
const _cm = new THREE.Matrix4()
const _cq = new THREE.Quaternion()
let cloudIdx = 0

for (let c = 0; c < cloudDefs.length; c++) {
  const cd = cloudDefs[c]
  const s = cd.s
  const indices = []
  for (const sub of cloudSubParts) {
    _cm.compose(
      new THREE.Vector3(cd.x + sub.ox * s, cd.y + sub.oy * s, cd.z + sub.oz * s),
      _cq,
      new THREE.Vector3(sub.sx * s, sub.sy * s, sub.sz * s)
    )
    cloudInstancedMesh.setMatrixAt(cloudIdx, _cm)
    indices.push(cloudIdx)
    cloudIdx++
  }
  cloudInstanceData.push({
    baseX: cd.x,
    baseY: cd.y-6,
    baseZ: cd.z,
    speed: 0.15 + Math.random() * 0.2,
    drift: Math.random() * Math.PI * 2,
    indices,
    s,
  })
}
cloudInstancedMesh.instanceMatrix.needsUpdate = true
cloudInstancedMesh.frustumCulled = false // Add this line
scene.add(cloudInstancedMesh)

// ─── Sea floaters (instanced) ───────────────────────────────────────────────
const seaFloaterCount = 12
const seaFloaterMesh = new THREE.InstancedMesh(sphereGeo, accentMats[0], seaFloaterCount)
seaFloaterMesh.castShadow = false
seaFloaterMesh.receiveShadow = false

// We need per-instance color since each floater has a different accent color
// InstancedMesh supports per-instance color via instanceColor
const seaFloaterColors = new Float32Array(seaFloaterCount * 3)
const seaFloaterData = []
const _fm = new THREE.Matrix4()
const _fc = new THREE.Color()

for (let i = 0; i < seaFloaterCount; i++) {
  const angle = (i / seaFloaterCount) * Math.PI * 2
  const r = 15 + Math.sin(i * 2.3) * 5
  const bx = Math.cos(angle) * r
  const bz = Math.sin(angle) * r - 3
  const s = 0.15 + Math.random() * 0.15
  const baseY = -0.1

  _fm.compose(
    new THREE.Vector3(bx, baseY, bz),
    _cq,
    new THREE.Vector3(s, s, s)
  )
  seaFloaterMesh.setMatrixAt(i, _fm)

  // Set color from accent mats
  const mat = accentMats[i % accentMats.length]
  _fc.set(mat.color)
  seaFloaterColors[i * 3] = _fc.r
  seaFloaterColors[i * 3 + 1] = _fc.g
  seaFloaterColors[i * 3 + 2] = _fc.b

  seaFloaterData.push({
    x: bx, z: bz, baseY, s,
    amplitude: 0.1 + Math.random() * 0.1,
    speed: 0.6 + Math.random() * 0.8,
    phase: Math.random() * 6,
  })
}
seaFloaterMesh.instanceColor = new THREE.InstancedBufferAttribute(seaFloaterColors, 3)
seaFloaterMesh.instanceMatrix.needsUpdate = true
scene.add(seaFloaterMesh)

// ─── Scroll-driven camera path ──────────────────────────────────────────────
const waypoints = [
  { pos: [0, 8, 18],   target: [0, 0.5, 6] },
  { pos: [-5, 5, 6],   target: [-7, 0.8, -2] },
  { pos: [6, 5, 5],    target: [9, 0.5, -1] },
  { pos: [0, 6, -4],   target: [0, 1.5, -12] },
  { pos: [-4, 5, -8],  target: [-6, 0.5, -13] },
]

// ─── Sections & scroll ──────────────────────────────────────────────────────
const allSections = document.querySelectorAll('.sections .section')
const finaleEl = document.querySelector('.finale')

let sectionTops = []
let finaleTop = 0

function updateSectionOffsets() {
  const scrollY = window.scrollY
  sectionTops = Array.from(allSections).map(el => el.getBoundingClientRect().top + scrollY)
  finaleTop = finaleEl.getBoundingClientRect().top + scrollY
}
updateSectionOffsets()

// Smooth camera interpolation state
let currentWaypoint = 0
let targetWaypoint = 0
let camTransitionStart = 0
let camTransitionProgress = 1
const camPos = new THREE.Vector3().set(...waypoints[0].pos)
const camTarget = new THREE.Vector3().set(...waypoints[0].target)
const camPosFrom = new THREE.Vector3()
const camPosTo = new THREE.Vector3()
const camTargetFrom = new THREE.Vector3()
const camTargetTo = new THREE.Vector3()

const camPosGoal = new THREE.Vector3().set(...waypoints[0].pos)
const camTargetGoal = new THREE.Vector3().set(...waypoints[0].target)

function ease(t) {
  return (easings[params.cameraEase] || easings.cubicInOut)(t)
}

function updateCameraProgress() {
  const viewportCenter = window.scrollY + innerHeight / 2

  let newWP = 0
  for (let i = sectionTops.length - 1; i >= 0; i--) {
    if (viewportCenter >= sectionTops[i]) {
      newWP = Math.min(i + 1, waypoints.length - 1)
      break
    }
  }

  if (newWP !== targetWaypoint) {
    // Capture current un-swayed position to avoid sway offset leaking into the transition start
    camPosFrom.copy(camPosGoal).lerp(camPos, 1) // use where camPos actually is
    camPosFrom.copy(camPos)
    camTargetFrom.copy(camTarget)
    camPosTo.set(...waypoints[newWP].pos)
    camTargetTo.set(...waypoints[newWP].target)
    currentWaypoint = targetWaypoint
    targetWaypoint = newWP
    camTransitionStart = performance.now() / 1000
    camTransitionProgress = 0
  }
}

// Blur transition
let blurFrom = 0, blurTo = 0, blurTransitionStart = 0, blurTransitionProgress = 1

function updateBlurTarget(tb) {
  if (tb === blurTo) return
  blurFrom = params.blur
  blurTo = tb
  blurTransitionStart = performance.now() / 1000
  blurTransitionProgress = 0
}

function onScroll() {
  updateCameraProgress()
  const vc = window.scrollY + innerHeight / 2
  updateBlurTarget(vc >= finaleTop ? 0.08 : 0)
}
window.addEventListener('scroll', onScroll, { passive: true })

// ─── Debug ──────────────────────────────────────────────────────────────────
function setDebug(enabled) {
  params.debug = enabled
  controls.enabled = enabled
  stats.dom.style.display = enabled ? '' : 'none'
  debugOverlay.style.display = enabled ? 'block' : 'none'
}
setDebug(params.debug)

window.addEventListener('keydown', (e) => {
  if (e.key === 'p' || e.key === 'P') setDebug(!params.debug)
})

// ─── Resize ─────────────────────────────────────────────────────────────────
window.addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight
  camera.updateProjectionMatrix()
  renderer.setSize(innerWidth, innerHeight)
  updateSectionOffsets()
  updateCameraProgress()
})

// ─── Animate ────────────────────────────────────────────────────────────────
const clock = new THREE.Clock()
const colliderPos = new THREE.Vector3(0, 5, 0) // reusable collider position

async function animate() {
  const dt = Math.min(clock.getDelta(), 0.05)
  const t = clock.elapsedTime

  controls.update()

  // Smooth camera transitions
  if (!params.debug) {
    if (camTransitionProgress < 1) {
      camTransitionProgress = Math.min((performance.now() / 1000 - camTransitionStart) / params.cameraTransitionDuration, 1)
      const e = ease(camTransitionProgress)
      camPos.lerpVectors(camPosFrom, camPosTo, e)
      camTarget.lerpVectors(camTargetFrom, camTargetTo, e)
    } else {
      // Only apply chase lerp when no transition is active — prevents fight between ease and chase
      const wp = waypoints[targetWaypoint]
      const goalPos = camPosTo.set(wp.pos[0], wp.pos[1], wp.pos[2])
      const goalTarget = camTargetTo.set(wp.target[0], wp.target[1], wp.target[2])
      const chaseRate = 3
      const chaseFactor = 1 - Math.exp(-chaseRate * dt)
      camPos.lerp(goalPos, chaseFactor)
      camTarget.lerp(goalTarget, chaseFactor)
    }

    const swayX = Math.sin(t * 0.3) * 0.12
    const swayY = Math.cos(t * 0.25) * 0.06
    camera.position.set(camPos.x + swayX, camPos.y + swayY, camPos.z)
    camera.lookAt(camTarget.x, camTarget.y, camTarget.z)
  }

  // Blur transition
  if (blurTransitionProgress < 1) {
    blurTransitionProgress = Math.min((performance.now() / 1000 - blurTransitionStart) / 0.3, 1)
    const e = easings.quadOut(blurTransitionProgress)
    params.blur = blurFrom + (blurTo - blurFrom) * e
    blurDirectionU.value = params.blur * 10
    // Toggle blur pass on/off
    setBlurActive(params.blur > 0.001)
  }

  // Bouncing objects
  for (const b of bouncingObjects) {
    b.mesh.position.y = b.baseY + Math.sin(t * b.speed + b.phase) * b.amplitude
  }

  for (const haBalloon of hotAirBalloons) {
    const angle = t * haBalloon.speed + haBalloon.phase
    haBalloon.mesh.position.set(
      haBalloon.centerX + Math.sin(angle) * haBalloon.radiusX,
      haBalloon.baseY + Math.sin(t * haBalloon.bobSpeed + haBalloon.phase) * haBalloon.bobAmp,
      haBalloon.centerZ + Math.cos(angle) * haBalloon.radiusZ
    )
    haBalloon.mesh.rotation.z = Math.sin(t * 1.2 + haBalloon.phase) * 0.05
    haBalloon.mesh.rotation.x = Math.cos(t * 0.8 + haBalloon.phase) * 0.03
    haBalloon.mesh.rotation.y = -angle + Math.PI * 0.5
  }

  const turtlePhase = t * 0.07
  const turtleX = Math.cos(turtlePhase) * 8.0
  // Island 1: radius=4, scaleX=1.5, scaleZ=0.8, topMesh scaleY = 1.5/4*0.7 = 0.2625
  // Turtle walks at world Z=8.5 → island-local z=0.5; precompute z term: (0.5/0.8)^2 = 0.39
  const turtleIslandSq = 16 - (turtleX / 1.5) ** 2 - 0.39
  const turtleBaseY = turtleIslandSq > 0
    ? 0.2625 * Math.sqrt(turtleIslandSq) + 0.05
    : 0
  const turtleSwimTilt = Math.max(0, Math.min(1, (1 - turtleIslandSq) / 5))
  turtle.mesh.position.set(
    turtleX,
    turtleBaseY + Math.sin(t * 5) * 0.012,
    8.5 + Math.sin(t * 2.3) * 0.12
  )
  turtle.mesh.rotation.x = -0.45 * turtleSwimTilt
  turtle.mesh.rotation.y = -(Math.PI / 2) * Math.tanh(Math.sin(turtlePhase) * 4)
  turtle.neck.rotation.x = Math.PI / 2 + Math.sin(t * 2.2) * 0.06
  turtle.head.rotation.x = Math.sin(t * 2.2) * 0.08
  turtle.legs.forEach((leg, index) => {
    const stride = Math.sin(t * 5 + index * Math.PI * 0.5) * 0.045
    leg.position.y = -0.15 + stride
    leg.rotation.z = stride * (index < 2 ? 4.6 : 3.2)
  })

  seahorse.mesh.position.set(
    2.4 + Math.sin(t * 0.7) * 0.35,
    -0.18 + Math.sin(t * 2.4) * 0.12,
    12.4 + Math.cos(t * 0.9) * 0.4
  )
  seahorse.mesh.rotation.y = -0.5 + Math.sin(t * 0.7) * 0.18
  seahorse.mesh.rotation.z = Math.sin(t * 1.6) * 0.08
  seahorse.head.rotation.z = Math.sin(t * 1.6 + 0.5) * 0.12
  seahorse.fin.rotation.y = Math.sin(t * 6) * 0.5
  seahorse.tail.rotation.z = -0.55 + Math.sin(t * 2.8) * 0.18
  seahorse.tailSegments.forEach((segment, index) => {
    segment.position.x = -0.08 * index + Math.sin(t * 2.8 + index * 0.5) * 0.02 * (index + 1)
  })
/*
  dolphin.mesh.position.set(
    2.2 - Math.sin(t * 0.55) * 1.5,
    -0.08 + Math.sin(t * 1.6) * 0.07,
    12.15 + Math.cos(t * 0.55) * 0.25
  )
  dolphin.mesh.rotation.y = -Math.PI + Math.cos(t * 0.55) * 0.12
  dolphin.mesh.rotation.x = 0
  dolphin.mesh.rotation.z = 0
  dolphin.tail.rotation.y = Math.sin(t * 6.5) * 0.45
  dolphin.dorsalFin.rotation.x = Math.sin(t * 1.2) * 0.05
  dolphin.sideFins.forEach((fin, index) => {
    fin.rotation.y = (index === 0 ? 1 : -1) * Math.sin(t * 3.4) * 0.12
  })
    */

  // Drifting clouds (instanced)
  for (const c of cloudInstanceData) {
    const dx = Math.sin(t * c.speed + c.drift) * 1.5
    const dy = Math.sin(t * c.speed * 0.7 + c.drift + 1) * 0.3
    for (let j = 0; j < c.indices.length; j++) {
      const sub = cloudSubParts[j]
      const idx = c.indices[j]
      _cm.compose(
        new THREE.Vector3(
          c.baseX + sub.ox * c.s + dx,
          c.baseY + sub.oy * c.s + dy,
          c.baseZ + sub.oz * c.s, // Use c.baseZ here
        ),
        _cq,
        new THREE.Vector3(sub.sx * c.s, sub.sy * c.s, sub.sz * c.s)
      )
      cloudInstancedMesh.setMatrixAt(idx, _cm)
    }
  }
  cloudInstancedMesh.instanceMatrix.needsUpdate = true

  // Sea floaters (instanced)
  for (let i = 0; i < seaFloaterCount; i++) {
    const sf = seaFloaterData[i]
    const y = sf.baseY + Math.sin(t * sf.speed + sf.phase) * sf.amplitude
    _fm.compose(
      new THREE.Vector3(sf.x, y, sf.z),
      _cq,
      new THREE.Vector3(sf.s, sf.s, sf.s)
    )
    seaFloaterMesh.setMatrixAt(i, _fm)
  }
  seaFloaterMesh.instanceMatrix.needsUpdate = true

  camera.updateMatrixWorld()

  if (!deferredInitDone) {
    deferredInit()
  }

  // Water compute — every frame for smooth visuals
  if (waterPlane) {
    waterPlane.update(mouseActive ? mouseNDC : mouseNDCIdle, camera, colliderPos, 0)
  }

  renderPipeline.render()
  stats.update()
}
renderer.setAnimationLoop(animate)
