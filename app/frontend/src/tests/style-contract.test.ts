import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

const root = process.cwd();
const styles = readFileSync(resolve(root, "src/styles.css"), "utf8");
const appSource = readFileSync(resolve(root, "src/App.tsx"), "utf8");

describe("style contract", () => {
  it("defines focus, responsive, reduced-motion and source styles", () => {
    expect(styles).toContain(":focus-visible");
    expect(styles).toContain("@media (max-width: 760px)");
    expect(styles).toContain("prefers-reduced-motion");
    expect(styles).toContain(".source-fresh");
    expect(styles).toContain(".source-legacy");
    expect(styles).toContain(".source-fixture");
  });

  it("does not load external images or the old frontend", () => {
    expect(styles).not.toContain("url(");
    expect(appSource).not.toContain("HetVis-main");
  });
});
