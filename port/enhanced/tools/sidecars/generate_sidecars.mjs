#!/usr/bin/env node
// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 the OpenRAC contributors
//
// Writes a texture pack's material maps (Materials/<hash>/*.png, docs/port/TEXTURE_PACKS.md) for
// the textures of exported levels, with the material-map pipeline of openrac-graphics-preparation
// (src/material-maps.js: de-lit albedo, height bands, normals, AO, roughness, emission...).
//
// What a texture is (lamp, metal, character...) comes from the level's own names first: the moby
// classes that use it (a "lightpost"'s textures glow where they are bright), a short list of the
// hero's textures (Clank's body is metal, his antenna bulb glows), and else the preparation's own
// classifier. The port reads the maps by the original texture's hash, so no colour texture changes.
//
//   node generate_sidecars.mjs --prep <openrac-graphics-preparation> \
//       --levels <assets/port/rac1/pal> --out <data/texture_packs/rac1-pal> [--level level_03]...

import {existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync} from "node:fs";
import {dirname, join, resolve} from "node:path";
import {pathToFileURL} from "node:url";
import {imageHash, readPng, writePng} from "./png.mjs";

const args = process.argv.slice(2);
const option = (name) => {
  const at = args.indexOf(name);
  return at >= 0 ? args[at + 1] : undefined;
};
const prep = option("--prep") ?? process.env.OPENRAC_GRAPHICS_PREPARATION;
const levelsRoot = option("--levels");
const out = option("--out");
const only = args.flatMap((a, i) => (a === "--level" ? [args[i + 1]] : []));
const force = args.includes("--force");
if (!prep || !levelsRoot || !out) {
  console.error("usage: generate_sidecars.mjs --prep DIR --levels DIR --out PACK [--level NAME]... [--force]");
  process.exit(2);
}
const {deriveMaterialMaps} = await import(pathToFileURL(resolve(prep, "src/material-maps.js")).href);
const {classifyMaterial} = await import(pathToFileURL(resolve(prep, "src/material-classify.js")).href);

// The hero's textures (the same in every level): Clank's body and eyes.
const HERO = {
  "7c432adc2d464f8e": "clank",     // body plates, rivets and the red antenna bulb
  bf5f81ccc718cc29: "clank_eyes", // eye glass
};
const LAMP = /(light|lamp|lantern|neon|bulb|sign|beacon|torch|glow)/;
const NOT_LAMP = /(lightning|lighter|flight|lights?_?off)/;

const lum = (r, g, b) => (54 * r + 183 * g + 19 * b) / 256 / 255;
const sat = (r, g, b) => {
  const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
  return mx ? (mx - mn) / mx : 0;
};
const smooth = (a, b, x) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};
const grey = (image, value) => {
  const data = new Uint8ClampedArray(image.width * image.height * 4);
  for (let i = 0; i < data.length; i += 4) {
    data[i] = data[i + 1] = data[i + 2] = value(i);
    data[i + 3] = 255;
  }
  return {width: image.width, height: image.height, data};
};

function glbJson(path) {
  const data = readFileSync(path);
  const length = data.readUInt32LE(12);
  return JSON.parse(data.toString("utf8", 20, 20 + length));
}

// Every texture of a level and the names of what uses it.
function levelTextures(level) {
  const manifest = JSON.parse(readFileSync(join(level, "manifest.json"), "utf8"));
  const hints = new Map();
  const note = (glb, hint) => {
    const path = join(level, glb);
    if (!existsSync(path)) return;
    for (const image of glbJson(path).images ?? []) {
      if (!image.uri) continue;
      const texture = resolve(dirname(path), image.uri);
      if (!hints.has(texture)) hints.set(texture, new Set());
      hints.get(texture).add(hint);
    }
  };
  for (const [kind, classes] of Object.entries(manifest.classes ?? {})) {
    for (const [id, info] of Object.entries(classes)) {
      if (info?.mesh) note(info.mesh, `${kind}/${info.name ?? id}`);
    }
  }
  if (typeof manifest.terrain === "string") note(manifest.terrain, "terrain");
  else if (manifest.terrain?.mesh) note(manifest.terrain.mesh, "terrain");
  const textures = (manifest.textures ?? []).map((t) => resolve(level, t));
  return textures.map((texture) => ({texture, hints: [...(hints.get(texture) ?? [])]}));
}

// What a texture is, for the preparation's pipeline and for the passes after it.
function kindOf(hash, hints, image) {
  if (HERO[hash]) return {kind: HERO[hash], material: HERO[hash] === "clank" ? "metal" : "energy"};
  const names = hints.map((h) => h.toLowerCase());
  if (names.some((n) => LAMP.test(n) && !NOT_LAMP.test(n))) return {kind: "lamp", material: "generic"};
  if (names.some((n) => n === "moby/ratchet")) return {kind: "character", material: "character"};
  const textureClass = names.every((n) => n.startsWith("sky")) && names.length ? "sky" : "diffuse";
  const found = classifyMaterial(names, image, textureClass);
  return {kind: found.material, material: found.material, confidence: found.confidence};
}

function derive(image, {kind, material, confidence = 1}) {
  const maps = deriveMaterialMaps(image, material, {curvature: false});
  const {width, height, data} = image;
  const emissive = {width, height, data: new Uint8ClampedArray(width * height * 4)};
  let glowing = 0;
  const metallic = grey(image, (i) => {
    const r = data[i], g = data[i + 1], b = data[i + 2];
    // Paint is no metal: a saturated texel stays dielectric whatever the surface is.
    const bare = 1 - smooth(0.3, 0.55, sat(r, g, b));
    if (kind === "clank") return 242 * bare;
    if (material === "metal") return 255 * (confidence >= 0.9 ? 0.9 : 0.55) * bare;
    if (material === "weapon") return 255 * 0.6 * bare;
    return 0;
  });
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i], g = data[i + 1], b = data[i + 2], a = data[i + 3];
    emissive.data[i + 3] = 255;
    if (a < 16) continue;
    let e = 0;
    if (kind === "lamp") {
      // A lamp's bright parts are its light; its posts, shades and polished plates are not: a
      // near-white texel, or a strongly coloured one at full brightness (a red sign bulb).
      const peak = Math.max(r, g, b) / 255;
      e = Math.max(smooth(0.72, 0.92, lum(r, g, b)),
                   smooth(0.78, 0.96, peak) * smooth(0.45, 0.7, sat(r, g, b)));
    } else if (kind === "clank") {
      // The antenna bulb: the strongly saturated, bright red.
      e = smooth(0.45, 0.7, sat(r, g, b)) * smooth(0.25, 0.5, Math.max(r, g, b) / 255);
    } else if (kind === "clank_eyes" || material === "energy") {
      e = smooth(0.5, 0.8, Math.max(lum(r, g, b), (Math.max(r, g, b) / 255) * 0.8));
    } else {
      // The preparation's own conservative mask (bright, emissive-looking texels only).
      // Only its strong texels: a faint mask would make every highlight of a texture glow.
      const p = maps.emissive.data;
      e = Math.max(p[i], p[i + 1], p[i + 2]) / Math.max(1, Math.max(r, g, b));
      e = e >= 0.45 ? e : 0;
    }
    if (e > 0.05) {
      emissive.data[i] = r * e;
      emissive.data[i + 1] = g * e;
      emissive.data[i + 2] = b * e;
      glowing++;
    }
  }
  const result = {
    normal: maps.normal,
    roughness: maps.roughness,
    ao: maps.ao,
    height: maps.height,
    cavity: maps.cavity,
    edgewear: maps.edgewear,
  };
  if (glowing > 0) result.emissive = emissive;
  if (metallic.data.some((v, i) => i % 4 === 0 && v > 8)) {
    result.metallic = metallic;
    // Metal's F0 is its colour (mesh.frag); the map marks it as reflective at all.
    result.specular = grey(image, (i) => (metallic.data[i] > 8 ? 230 : 10));
  }
  return result;
}

const levels = readdirSync(levelsRoot)
  .filter((name) => name.startsWith("level_") && (only.length === 0 || only.includes(name)))
  .sort();
const done = new Map();
let written = 0, lamps = 0, metals = 0, glows = 0;
for (const name of levels) {
  const level = join(levelsRoot, name);
  if (!existsSync(join(level, "manifest.json"))) continue;
  for (const {texture, hints} of levelTextures(level)) {
    if (!existsSync(texture)) continue;
    let image;
    try {
      image = readPng(texture);
    } catch (error) {
      console.warn(`${texture}: ${error.message}`);
      continue;
    }
    const hash = imageHash(image);
    const known = done.get(hash);
    // A texture several levels share: the most telling names win (a lamp anywhere is a lamp).
    const kind = kindOf(hash, hints, image);
    if (known && !(kind.kind === "lamp" && known.kind !== "lamp")) continue;
    done.set(hash, kind);
    const folder = join(out, "Materials", hash);
    if (!force && existsSync(join(folder, "normal.png")) && known === undefined
        && existsSync(join(folder, ".kind")) && readFileSync(join(folder, ".kind"), "utf8") === kind.kind) {
      continue;
    }
    if (kind.material === "sky" || kind.material === "ui") continue;
    const maps = derive(image, kind);
    mkdirSync(folder, {recursive: true});
    for (const [map, picture] of Object.entries(maps)) writePng(join(folder, `${map}.png`), picture);
    writeFileSync(join(folder, ".kind"), kind.kind);
    written++;
    lamps += kind.kind === "lamp" ? 1 : 0;
    metals += maps.metallic ? 1 : 0;
    glows += maps.emissive ? 1 : 0;
  }
  console.log(`${name}: ${done.size} textures so far`);
}
console.log(`${written} material set(s) written to ${join(out, "Materials")}: ${lamps} lamp texture(s), ${glows} glow, ${metals} metal`);
