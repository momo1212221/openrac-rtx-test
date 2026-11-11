// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 the OpenRAC contributors
//
// A small PNG reader and writer for the sidecar generator: 8-bit greyscale, grey+alpha, RGB, RGBA
// and palette images (with tRNS), non-interlaced, read as RGBA8 the way stb_image's 4-channel load
// gives them to the port (so that the hashes agree with renderer/texture_pack.cpp's image_hash).

import {readFileSync, writeFileSync} from "node:fs";
import {deflateSync, inflateSync} from "node:zlib";

const SIGNATURE = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);

export function readPng(path) {
  const data = readFileSync(path);
  if (!data.subarray(0, 8).equals(SIGNATURE)) throw new Error(`${path}: not a PNG`);
  let at = 8, width = 0, height = 0, depth = 0, type = 0, interlace = 0;
  let palette = null, transparency = null;
  const idat = [];
  while (at < data.length) {
    const length = data.readUInt32BE(at), kind = data.toString("latin1", at + 4, at + 8);
    const body = data.subarray(at + 8, at + 8 + length);
    if (kind === "IHDR") {
      width = body.readUInt32BE(0); height = body.readUInt32BE(4);
      depth = body[8]; type = body[9]; interlace = body[12];
    } else if (kind === "PLTE") palette = body;
    else if (kind === "tRNS") transparency = body;
    else if (kind === "IDAT") idat.push(body);
    else if (kind === "IEND") break;
    at += 12 + length;
  }
  if (depth !== 8 || interlace !== 0) throw new Error(`${path}: ${depth}-bit/interlaced PNGs are not read`);
  const channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[type];
  if (!channels) throw new Error(`${path}: colour type ${type} is not read`);
  const raw = inflateSync(Buffer.concat(idat));
  const stride = width * channels;
  const rows = Buffer.alloc(stride * height);
  for (let y = 0; y < height; y++) {
    const filter = raw[y * (stride + 1)];
    const line = raw.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1));
    for (let x = 0; x < stride; x++) {
      const a = x >= channels ? rows[y * stride + x - channels] : 0;
      const b = y > 0 ? rows[(y - 1) * stride + x] : 0;
      const c = x >= channels && y > 0 ? rows[(y - 1) * stride + x - channels] : 0;
      let v = line[x];
      if (filter === 1) v += a;
      else if (filter === 2) v += b;
      else if (filter === 3) v += (a + b) >> 1;
      else if (filter === 4) {
        const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
        v += pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
      }
      rows[y * stride + x] = v & 255;
    }
  }
  const out = new Uint8ClampedArray(width * height * 4);
  for (let p = 0; p < width * height; p++) {
    const s = p * channels, d = p * 4;
    if (type === 0) { out[d] = out[d + 1] = out[d + 2] = rows[s]; out[d + 3] = 255; }
    else if (type === 4) { out[d] = out[d + 1] = out[d + 2] = rows[s]; out[d + 3] = rows[s + 1]; }
    else if (type === 2) { out[d] = rows[s]; out[d + 1] = rows[s + 1]; out[d + 2] = rows[s + 2]; out[d + 3] = 255; }
    else if (type === 6) { for (let k = 0; k < 4; k++) out[d + k] = rows[s + k]; }
    else {
      const i = rows[s];
      out[d] = palette[i * 3]; out[d + 1] = palette[i * 3 + 1]; out[d + 2] = palette[i * 3 + 2];
      out[d + 3] = transparency && i < transparency.length ? transparency[i] : 255;
    }
  }
  return {width, height, data: out};
}

const CRC = new Int32Array(256).map((_, n) => {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  return c;
});
function crc32(buffer) {
  let c = -1;
  for (const byte of buffer) c = CRC[(c ^ byte) & 255] ^ (c >>> 8);
  return (c ^ -1) >>> 0;
}
function chunk(kind, body) {
  const head = Buffer.alloc(8);
  head.writeUInt32BE(body.length, 0);
  head.write(kind, 4, "latin1");
  const tail = Buffer.alloc(4);
  tail.writeUInt32BE(crc32(Buffer.concat([head.subarray(4), body])), 0);
  return Buffer.concat([head, body, tail]);
}

export function writePng(path, {width, height, data}) {
  const header = Buffer.alloc(13);
  header.writeUInt32BE(width, 0); header.writeUInt32BE(height, 4);
  header[8] = 8; header[9] = 6;
  const raw = Buffer.alloc((width * 4 + 1) * height);
  for (let y = 0; y < height; y++) {
    raw[y * (width * 4 + 1)] = 0;
    Buffer.from(data.buffer, data.byteOffset + y * width * 4, width * 4).copy(raw, y * (width * 4 + 1) + 1);
  }
  writeFileSync(path, Buffer.concat([
    SIGNATURE, chunk("IHDR", header), chunk("IDAT", deflateSync(raw)), chunk("IEND", Buffer.alloc(0)),
  ]));
}

// renderer/texture_pack.cpp's image_hash: FNV-1a over the width and height (4 bytes each, little
// endian) and then the RGBA8 pixels.
export function imageHash({width, height, data}) {
  // The 64-bit state as two 32-bit halves; the prime is 2^40 + 0x1b3. The port's offset basis is
  // 1469598103934665603 (not FNV's usual 14695981039346656037), and the names must match it.
  let hi = 0x14650fb0, lo = 0x739d0383;
  const mix = (byte) => {
    lo = (lo ^ byte) >>> 0;
    const low = lo * 0x1b3;
    const carry = Math.floor(low / 4294967296);
    hi = (hi * 0x1b3 + ((lo << 8) >>> 0) + carry) >>> 0;
    lo = low >>> 0;
  };
  for (const side of [width, height]) for (let i = 0; i < 4; i++) mix((side >>> (8 * i)) & 255);
  for (let i = 0; i < data.length; i++) mix(data[i]);
  return hi.toString(16).padStart(8, "0") + lo.toString(16).padStart(8, "0");
}
