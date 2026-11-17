"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const parser = require(path.resolve(__dirname, "../../docs/_static/dxf_geometry_import.js"));

function dxf(entities) {
  return [0, "SECTION", 2, "ENTITIES", ...entities, 0, "ENDSEC", 0, "EOF"].join("\n") + "\n";
}

function lw(points, closed = true) {
  return [0, "LWPOLYLINE", 70, closed ? 1 : 0,
    ...points.flatMap(([x, y, bulge]) => bulge === undefined
      ? [10, x, 20, y] : [10, x, 20, y, 42, bulge])];
}

const rectangle = lw([[0, 0], [4, 0], [4, 3], [0, 3]]);
const hole = lw([[1, 1], [2, 1], [2, 2], [1, 2]]);
const [outer, holes] = parser.parse(dxf([...hole, ...rectangle]));
assert.equal(outer.length, 4);
assert.equal(holes.length, 1);
assert.deepEqual(outer[0], [0, 0]);
assert.deepEqual(holes[0][0], [1, 1]);
const example = fs.readFileSync(path.resolve(__dirname, "../../examples/geometries/rectangle_with_hole.dxf"), "utf8");
assert.equal(parser.parse(example)[1].length, 1);
const comsolExample = fs.readFileSync(path.resolve(__dirname, "../../examples/geometries/comsol_2d_export_example.dxf"), "utf8");
const [comsolOuter, comsolHoles] = parser.parse(comsolExample);
assert.equal(comsolOuter.length, 4);
assert.equal(comsolHoles.length, 1);
assert.equal(comsolHoles[0].length, 8);
assert.equal(parser.parseDetailed(comsolExample).unitsCode, 6);
assert.deepEqual(parser.parseDetailed(comsolExample, {unitScale: 0.001}).outer[1], [0.008, 0]);
assert.throws(() => parser.parseDetailed(comsolExample, {unitScale: 0}), /正数/);
const cadExample = fs.readFileSync(path.resolve(__dirname, "../../examples/geometries/cad_l_shape_example.dxf"), "utf8");
const [cadOuter, cadHoles] = parser.parse(cadExample);
assert.equal(cadOuter.length, 6);
assert.equal(cadHoles.length, 0);

const [uniqueOuter, uniqueHoles] = parser.parse(dxf([...rectangle, ...rectangle]));
assert.equal(uniqueOuter.length, 4);
assert.equal(uniqueHoles.length, 0);

const [legacy] = parser.parse(dxf([
  0, "POLYLINE", 70, 1,
  0, "VERTEX", 10, 0, 20, 0,
  0, "VERTEX", 10, 2, 20, 0,
  0, "VERTEX", 10, 0, 20, 2,
  0, "SEQEND",
]));
assert.equal(legacy.length, 3);

const [circle] = parser.parse(dxf([0, "CIRCLE", 10, 5, 20, 2, 40, 3]));
assert.equal(circle.length, 128);
assert.ok(Math.abs(circle[0][0] - 8) < 1e-8);

const [ellipse] = parser.parse(dxf([
  0, "ELLIPSE", 10, 0, 20, 0, 11, 2, 21, 0, 40, 0.5,
]));
assert.equal(ellipse.length, 128);
assert.ok(Math.abs(ellipse[32][1] - 1) < 1e-8);

const [halfEllipse] = parser.parse(dxf([
  0, "ELLIPSE", 10, 0, 20, 0, 11, 2, 21, 0, 40, 0.5, 41, 0, 42, Math.PI,
  0, "LINE", 10, -2, 20, 0, 11, 2, 21, 0,
]));
assert.ok(halfEllipse.length > 30);

const [bulged] = parser.parse(dxf(lw([[0, 0, 1], [2, 0], [1, -2]])));
assert.ok(bulged.length > 3);

const quarterBulge = -(Math.SQRT2 - 1);
const comsolCircleHoles = parser.parseDetailed(dxf([
  ...rectangle,
  ...lw([[1, 1.5, quarterBulge], [1.5, 2, quarterBulge],
    [2, 1.5, quarterBulge], [1.5, 1, quarterBulge]]),
]), {unitScale: 2});
assert.equal(comsolCircleHoles.circularHoles.length, 1);
assert.deepEqual(comsolCircleHoles.circularHoles[0].center, [3, 3]);
assert.ok(Math.abs(comsolCircleHoles.circularHoles[0].radius - 1) < 1e-9);
assert.ok(comsolCircleHoles.holes[0].length >= 72);
const originalCircle = comsolCircleHoles.circularHoles[0];
const firstAngle = Math.atan2(comsolCircleHoles.holes[0][0][1] - originalCircle.center[1],
  comsolCircleHoles.holes[0][0][0] - originalCircle.center[0]);
const editedCircleRing = comsolCircleHoles.holes[0].map((_point, index, ring) => [
  3.25 + 1.5 * Math.cos(firstAngle - 2 * Math.PI * index / ring.length),
  2.75 + 1.5 * Math.sin(firstAngle - 2 * Math.PI * index / ring.length),
]);
assert.deepEqual(parser.circularRing(editedCircleRing).center, [3.25, 2.75]);
assert.equal(parser.circularRing(editedCircleRing).radius, 1.5);
assert.equal(parser.parseDetailed(dxf([...rectangle, ...hole])).circularHoles[0], null);

const line = (x1, y1, x2, y2) => [0, "LINE", 10, x1, 20, y1, 11, x2, 21, y2];
const shuffledLines = dxf([
  ...line(4, 3, 0, 3), ...line(0, 0, 4, 0),
  ...line(0, 3, 0, 0), ...line(4, 0, 4, 3),
]);
const stitched = parser.parseDetailed(shuffledLines);
assert.equal(stitched.outer.length, 4);
assert.equal(stitched.stitchedEntityCount, 4);
assert.equal(stitched.holes.length, 0);

const [mixedArc] = parser.parse(dxf([
  ...line(-1, 0, 1, 0),
  0, "ARC", 10, 0, 20, 0, 40, 1, 50, 0, 51, 180,
]));
assert.ok(mixedArc.length > 20);

const [mixedSpline] = parser.parse(dxf([
  0, "SPLINE", 70, 0, 71, 2, 72, 6, 73, 3,
  40, 0, 40, 0, 40, 0, 40, 1, 40, 1, 40, 1,
  10, 0, 20, 0, 10, 1, 20, 2, 10, 2, 20, 0,
  ...line(2, 0, 0, 0),
]));
assert.ok(mixedSpline.length >= 64);
assert.ok(mixedSpline.some(([x, y]) => Math.abs(x - 1) < 1e-8 && Math.abs(y - 1) < 1e-8));

const [openPolylineStitched] = parser.parse(dxf([
  ...lw([[0, 0], [2, 0], [2, 2]], false),
  ...line(2, 2, 0, 2), ...line(0, 2, 0, 0),
]));
assert.equal(openPolylineStitched.length, 4);

assert.throws(() => parser.parse(dxf(lw([[0, 0], [1, 0], [0, 1]], false))), /无缺口|无分叉/);
assert.throws(() => parser.parse(dxf([...line(0, 0, 1, 0), ...line(1, 0, 0, 0), ...line(0, 0, 0, 1)])), /无缺口|无分叉/);
assert.throws(() => parser.parse(dxf([...rectangle, ...lw([[8, 0], [9, 0], [8, 1]])])), /互不相连/);
assert.throws(() => parser.parse(dxf([0, "SPLINE", 10, 0, 20, 0])), /SPLINE/);
assert.throws(() => parser.parse("not a dxf"), /ASCII DXF|组码/);

process.stdout.write("DXF_GEOMETRY_IMPORT_OK\n");
