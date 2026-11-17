"use strict";

const assert = require("node:assert/strict");
const path = require("node:path");
const tracer = require(path.resolve(__dirname, "../../docs/_static/polygon_image_trace.js"));

function image(width, height, background) {
  const rgba = new Uint8ClampedArray(width * height * 4);
  for (let pixel = 0; pixel < width * height; pixel += 1) {
    const offset = pixel * 4;
    rgba[offset] = background;
    rgba[offset + 1] = background;
    rgba[offset + 2] = background;
    rgba[offset + 3] = 255;
  }
  return rgba;
}

function paint(rgba, width, x0, x1, y0, y1, value) {
  for (let y = y0; y < y1; y += 1) {
    for (let x = x0; x < x1; x += 1) {
      const offset = 4 * (y * width + x);
      rgba[offset] = value;
      rgba[offset + 1] = value;
      rgba[offset + 2] = value;
    }
  }
}

function paintColour(rgba, width, x0, x1, y0, y1, colour) {
  for (let y = y0; y < y1; y += 1) {
    for (let x = x0; x < x1; x += 1) {
      const offset = 4 * (y * width + x);
      rgba.set(colour, offset);
    }
  }
}

function pointInPolygon(point, polygon) {
  let inside = false;
  for (let index = 0, previous = polygon.length - 1; index < polygon.length; previous = index++) {
    const currentPoint = polygon[index];
    const previousPoint = polygon[previous];
    if ((currentPoint[1] > point[1]) !== (previousPoint[1] > point[1])) {
      const crossing = currentPoint[0] + (point[1] - currentPoint[1]) *
        (previousPoint[0] - currentPoint[0]) / (previousPoint[1] - currentPoint[1]);
      if (point[0] < crossing) inside = !inside;
    }
  }
  return inside;
}

function paintPolygon(rgba, width, height, polygon, value) {
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if (!pointInPolygon([x + 0.5, y + 0.5], polygon)) continue;
      const offset = 4 * (y * width + x);
      rgba[offset] = value;
      rgba[offset + 1] = value;
      rgba[offset + 2] = value;
    }
  }
}

function paintImplicit(rgba, width, height, predicate, value) {
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if (!predicate(x + 0.5, y + 0.5)) continue;
      const offset = 4 * (y * width + x);
      rgba[offset] = value;
      rgba[offset + 1] = value;
      rgba[offset + 2] = value;
    }
  }
}

const width = 16;
const height = 12;
const rgba = image(width, height, 255);
paint(rgba, width, 1, 15, 1, 11, 0);
paint(rgba, width, 6, 10, 4, 8, 255);
paint(rgba, width, 0, 1, 0, 1, 0); // disconnected noise must be ignored

const polygon = tracer.imageDataToPolygon(rgba, width, height, {
  foreground: "auto",
  threshold: 128,
  bounds: [0, 8, -2, 4],
  simplifyTolerance: 0.25,
});

assert.equal(polygon.foreground, "dark");
assert.equal(polygon.holes.length, 1);
assert.ok(polygon.outer.length >= 4);
assert.ok(polygon.holes[0].length >= 4);
assert.ok(tracer.signedArea(polygon.outer) > 0);
assert.ok(tracer.signedArea(polygon.holes[0]) < 0);
assert.ok(polygon.outer.every((point) => point[0] >= 0 && point[0] <= 8));
assert.ok(polygon.outer.every((point) => point[1] >= -2 && point[1] <= 4));

const inverted = image(width, height, 0);
paint(inverted, width, 2, 14, 2, 10, 255);
const light = tracer.imageDataToPolygon(inverted, width, height, {
  foreground: "auto",
  threshold: 128,
});
assert.equal(light.foreground, "light");
assert.equal(light.holes.length, 0);
assert.equal(tracer.imageDataToPolygon(rgba, width, height, {
  foreground: "auto", threshold: "auto", simplifyTolerance: 0.25,
}).holes.length, 1);
assert.equal(tracer.imageDataToPolygon(inverted, width, height, {
  foreground: "auto", threshold: "auto", simplifyTolerance: 0.25,
}).foreground, "light");

// Grey filled soil with two white tunnel holes and black outlines. A fixed
// threshold of 128 sees only the outlines; automatic segmentation must include
// the grey fill and preserve both holes.
const tunnelSize = 256;
const tunnelRgba = image(tunnelSize, tunnelSize, 255);
paint(tunnelRgba, tunnelSize, 24, 232, 20, 236, 200);
paint(tunnelRgba, tunnelSize, 24, 232, 20, 22, 0);
paint(tunnelRgba, tunnelSize, 24, 232, 234, 236, 0);
paint(tunnelRgba, tunnelSize, 24, 26, 20, 236, 0);
paint(tunnelRgba, tunnelSize, 230, 232, 20, 236, 0);
for (const centerX of [76, 180]) {
  paintImplicit(tunnelRgba, tunnelSize, tunnelSize, (x, y) =>
    Math.hypot(x - centerX, y - 128) <= 22, 0);
  paintImplicit(tunnelRgba, tunnelSize, tunnelSize, (x, y) =>
    Math.hypot(x - centerX, y - 128) <= 20, 255);
}
const tracedTunnel = tracer.imageDataToPolygon(tunnelRgba, tunnelSize, tunnelSize, {
  foreground: "auto",
  threshold: "auto",
  simplifyTolerance: 2,
});
assert.equal(tracedTunnel.foreground, "dark");
assert.ok(tracedTunnel.threshold > 200 && tracedTunnel.threshold < 255);
assert.equal(tracedTunnel.outer.length, 4);
assert.equal(tracedTunnel.holes.length, 2);
assert.ok(tracedTunnel.holes.every((hole) => hole.length >= 8));
const fixedTunnel = tracer.imageDataToPolygon(tunnelRgba, tunnelSize, tunnelSize, {
  foreground: "auto",
  threshold: 128,
});
assert.notEqual(fixedTunnel.holes.length, 2);

// Colours can have almost identical grayscale luminance while being clearly
// distinguishable in RGB. The automatic path must not be luminance-only.
const colourSize = 80;
const colourRgba = new Uint8ClampedArray(colourSize * colourSize * 4);
paintColour(colourRgba, colourSize, 0, 80, 0, 80, [0, 128, 0, 255]);
paintColour(colourRgba, colourSize, 8, 72, 8, 72, [200, 68, 0, 255]);
paintColour(colourRgba, colourSize, 32, 48, 32, 48, [0, 128, 0, 255]);
const tracedColour = tracer.imageDataToPolygon(colourRgba, colourSize, colourSize);
assert.equal(tracedColour.segmentation, "color");
assert.equal(tracedColour.outer.length, 4);
assert.equal(tracedColour.holes.length, 1);

const transparentRgba = new Uint8ClampedArray(colourSize * colourSize * 4);
paintColour(transparentRgba, colourSize, 8, 72, 8, 72, [255, 255, 255, 255]);
paintColour(transparentRgba, colourSize, 32, 48, 32, 48, [0, 0, 0, 0]);
const tracedTransparent = tracer.imageDataToPolygon(transparentRgba, colourSize, colourSize);
assert.equal(tracedTransparent.segmentation, "alpha");
assert.equal(tracedTransparent.outer.length, 4);
assert.equal(tracedTransparent.holes.length, 1);

const square = [[[0, 0], [2, 0], [2, 2], [0, 2]]];
const stretchedEdge = tracer.setEdgeLength(square, 0, 0, 4);
assert.ok(Math.abs(stretchedEdge[0][1][0] - 4) < 1e-12);
assert.ok(Math.abs(stretchedEdge[0][1][1]) < 1e-12);

const changedAngle = tracer.setInteriorAngle(square, 0, 0, 45);
assert.ok(Math.abs(changedAngle[0][1][0] - Math.sqrt(2)) < 1e-12);
assert.ok(Math.abs(changedAngle[0][1][1] - Math.sqrt(2)) < 1e-12);

const transformed = tracer.transformRings(square, {
  angle: 90,
  scaleX: 2,
  scaleY: 1,
  translateX: 3,
  translateY: -2,
});
assert.ok(Math.abs(Math.abs(tracer.signedArea(transformed[0])) - 8) < 1e-10);

const squareEdges = tracer.edgeMeasurements(square[0]);
assert.deepEqual(squareEdges.map((edge) => [edge.startIndex, edge.endIndex]), [
  [0, 1], [1, 2], [2, 3], [3, 0],
]);
assert.deepEqual(squareEdges.map((edge) => edge.length), [2, 2, 2, 2]);

// CSS pixels are used by both PointerEvent and getBoundingClientRect.  The
// result therefore stays identical at 100%, 150% or 200% browser zoom.
const pointAt100 = tracer.clientPointToViewBox(
  410, 230,
  { left: 50, top: 50, width: 720, height: 360 },
  { x: 0, y: 0, width: 720, height: 360 }
);
const pointAt200 = tracer.clientPointToViewBox(
  820, 460,
  { left: 100, top: 100, width: 1440, height: 720 },
  { x: 0, y: 0, width: 720, height: 360 }
);
assert.deepEqual(pointAt100, [360, 180]);
assert.deepEqual(pointAt200, pointAt100);

// Default SVG preserveAspectRatio=xMidYMid meet may add letterboxing.  That
// margin must not shift the dragged data point.
const letterboxed = tracer.clientPointToViewBox(
  500, 300,
  { left: 100, top: 100, width: 800, height: 400 },
  { x: 0, y: 0, width: 400, height: 400 }
);
assert.deepEqual(letterboxed, [200, 200]);

// A five-point star has ten necessary boundary corners: five outer tips and
// five inner concave corners.  The 2 px default must remove raster stair-step
// points without turning the star into a pentagon.
const starSize = 256;
const starRgba = image(starSize, starSize, 255);
const star = Array.from({ length: 10 }, (_, index) => {
  const radius = index % 2 === 0 ? 92 : 39;
  const angle = -Math.PI / 2 + index * Math.PI / 5;
  return [128 + radius * Math.cos(angle), 128 + radius * Math.sin(angle)];
});
paintPolygon(starRgba, starSize, starSize, star, 0);
const tracedStar = tracer.imageDataToPolygon(starRgba, starSize, starSize, {
  foreground: "dark",
  threshold: 128,
  bounds: [-1, 1, -1, 1],
});
assert.equal(tracedStar.outer.length, 10);
assert.equal(tracedStar.holes.length, 0);
assert.equal(tracedStar.analyticShape, null);
assert.equal(tracedStar.curveShape, null);

// A raster circle must be promoted to an analytic disk instead of becoming
// dozens of short polygon edges.
const circleSize = 160;
const circleRgba = image(circleSize, circleSize, 255);
paintImplicit(circleRgba, circleSize, circleSize, (x, y) =>
  Math.hypot(x - 80, y - 80) <= 52, 0);
const tracedCircle = tracer.imageDataToPolygon(circleRgba, circleSize, circleSize, {
  foreground: "dark",
  threshold: 128,
  bounds: [-2, 2, -2, 2],
});
assert.equal(tracedCircle.analyticShape.type, "circle");
assert.equal(tracedCircle.curveShape.type, "closed_spline");
assert.ok(tracedCircle.analyticShape.confidence > 0.98);
assert.ok(Math.abs(tracedCircle.analyticShape.radius - 1.3) < 0.04);

// Axis-aligned rounded rectangles retain one exact corner-radius parameter.
const roundedWidth = 200;
const roundedHeight = 140;
const roundedRgba = image(roundedWidth, roundedHeight, 255);
const left = 25;
const right = 175;
const top = 20;
const bottom = 120;
const radius = 22;
paintImplicit(roundedRgba, roundedWidth, roundedHeight, (x, y) => {
  const centerX = (left + right) / 2;
  const centerY = (top + bottom) / 2;
  const qx = Math.abs(x - centerX) - ((right - left) / 2 - radius);
  const qy = Math.abs(y - centerY) - ((bottom - top) / 2 - radius);
  return Math.hypot(Math.max(qx, 0), Math.max(qy, 0)) + Math.min(Math.max(qx, qy), 0) <= radius;
}, 0);
const tracedRounded = tracer.imageDataToPolygon(roundedRgba, roundedWidth, roundedHeight, {
  foreground: "dark",
  threshold: 128,
  bounds: [0, 4, 0, 2.8],
});
assert.equal(tracedRounded.analyticShape.type, "rounded_rectangle");
assert.equal(tracedRounded.curveShape.type, "closed_spline");
assert.ok(tracedRounded.curveShape.controlPoints.length >= 8);
assert.ok(tracedRounded.analyticShape.confidence > 0.98);
assert.ok(Math.abs(tracedRounded.analyticShape.cornerRadius - 0.44) < 0.04);

console.log("POLYGON_IMAGE_TRACE_OK");
