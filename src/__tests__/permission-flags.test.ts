import { describe, expect, it } from "bun:test";

const { permissionFlags } = await import("../index.js");

describe("cursor permissionFlags", () => {
  it("fullAuto → --force (auto-approve all tools)", () => {
    expect(permissionFlags("fullAuto")).toEqual(["--force"]);
  });

  it("plan → no flag (print mode proposes edits only)", () => {
    expect(permissionFlags("plan")).toEqual([]);
  });

  it("acceptEdits → no flag", () => {
    expect(permissionFlags("acceptEdits")).toEqual([]);
  });

  it("undefined / unknown → acceptEdits (no --force)", () => {
    expect(permissionFlags(undefined)).toEqual([]);
    expect(permissionFlags("bogus" as never)).toEqual([]);
  });
});
