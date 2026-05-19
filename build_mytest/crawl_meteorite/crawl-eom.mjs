#!/usr/bin/env node
import { chromium } from "playwright";
import fs from "node:fs/promises";
import path from "node:path";

const DEFAULT_BASE = "https://encyclopedia-of-meteorites.com/";
const DEFAULT_OUT = "output/eom";
const DEFAULT_DELAY_MS = 1200;
const DEFAULT_MAX_PAGES = 20;
const DEFAULT_MAX_ITEMS = 200;

function parseArgs(argv) {
    const args = {
        baseUrl: DEFAULT_BASE,
        outDir: DEFAULT_OUT,
        delayMs: DEFAULT_DELAY_MS,
        maxPages: DEFAULT_MAX_PAGES,
        maxItems: DEFAULT_MAX_ITEMS,
        headless: true,
        resume: true,
        channel: process.env.EOM_BROWSER_CHANNEL || "msedge",
    };

    for (let i = 2; i < argv.length; i += 1) {
        const cur = argv[i];
        const next = () => argv[++i];
        if (cur === "--base") args.baseUrl = next();
        else if (cur === "--out") args.outDir = next();
        else if (cur === "--delay") args.delayMs = Number(next());
        else if (cur === "--max-pages") args.maxPages = Number(next());
        else if (cur === "--max-items") args.maxItems = Number(next());
        else if (cur === "--headed") args.headless = false;
        else if (cur === "--no-resume") args.resume = false;
        else if (cur === "--channel") args.channel = next();
        else if (cur === "--help" || cur === "-h") {
            console.log(`
Usage:
  node crawl-eom.mjs [options]

Options:
  --base <url>        Base URL, default ${DEFAULT_BASE}
  --out <dir>         Output directory, default ${DEFAULT_OUT}
  --delay <ms>        Delay between page visits, default ${DEFAULT_DELAY_MS}
  --max-pages <n>     Max listing pages to scan, default ${DEFAULT_MAX_PAGES}
  --max-items <n>     Max meteorite pages to process, default ${DEFAULT_MAX_ITEMS}
  --headed            Show the browser
  --no-resume         Ignore previous run state
  --channel <name>    Browser channel (default msedge)
`);
            process.exit(0);
        } else {
            throw new Error(`Unknown arg: ${cur}`);
        }
    }

    return args;
}

function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

async function ensureDir(dir) {
    await fs.mkdir(dir, { recursive: true });
}

function toAbs(base, href) {
    try {
        return new URL(href, base).href;
    } catch {
        return null;
    }
}

function safeName(input) {
    return (
        String(input)
            .replace(/[\\/:*?"<>|]+/g, "_")
            .replace(/\s+/g, " ")
            .trim()
            .slice(0, 180) || "item"
    );
}

function uniq(arr) {
    return [...new Set(arr)];
}

function looksLikeContentImage(img) {
    const src = img?.src || "";
    const file = src.toLowerCase();
    const pathname = (() => {
        try {
            return new URL(src).pathname.toLowerCase();
        } catch {
            return file;
        }
    })();
    const { width = 0, height = 0 } = img || {};
    if (!src) return false;
    if (
        /(ajax-loader|loader|spinner|logo|icon|flag|sprite|bullet|menu|arrow)/i.test(
            file,
        )
    )
        return false;
    if (!/\.(jpe?g|png|webp)$/i.test(pathname)) return false;
    if ((width > 0 || height > 0) && width < 80 && height < 80) return false;
    return true;
}

async function saveJson(file, value) {
    await fs.writeFile(file, JSON.stringify(value, null, 2));
}

async function readJson(file, fallback) {
    try {
        return JSON.parse(await fs.readFile(file, "utf8"));
    } catch {
        return fallback;
    }
}

async function main() {
    const opts = parseArgs(process.argv);
    const outDir = path.resolve(opts.outDir);
    const pagesDir = path.join(outDir, "pages");
    const imagesDir = path.join(outDir, "images");
    const stateFile = path.join(outDir, "state.json");
    const manifestFile = path.join(outDir, "manifest.json");

    await ensureDir(outDir);
    await ensureDir(pagesDir);
    await ensureDir(imagesDir);

    const state = opts.resume
        ? await readJson(stateFile, {
              visitedPages: [],
              visitedMeteorites: [],
              imageUrls: [],
          })
        : { visitedPages: [], visitedMeteorites: [], imageUrls: [] };
    state.visitedPages = state.visitedPages || [];
    state.visitedMeteorites = state.visitedMeteorites || [];
    state.imageUrls = state.imageUrls || [];

    const visitedPages = new Set(state.visitedPages);
    const visitedMeteorites = new Set(state.visitedMeteorites);
    const visitedImages = new Set(state.imageUrls);
    const records = [];

    const browser = await chromium.launch({
        headless: opts.headless,
        channel: opts.channel,
    });
    const context = await browser.newContext({
        userAgent:
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    });
    const page = await context.newPage();
    page.setDefaultTimeout(25000);
    page.setDefaultNavigationTimeout(90000);

    async function politeGoto(url) {
        for (let attempt = 1; attempt <= 2; attempt += 1) {
            try {
                await page.goto(url, {
                    waitUntil: "domcontentloaded",
                    timeout: 90000,
                });
                break;
            } catch (err) {
                if (attempt === 2) throw err;
                console.warn(`Retrying slow page: ${url}`);
                await sleep(opts.delayMs * 2);
            }
        }
        await page.waitForLoadState("load", { timeout: 30000 }).catch(() => {});
        await sleep(opts.delayMs);
    }

    async function extractMeteoriteLinks(listUrl) {
        await politeGoto(listUrl);
        if (/Meteorites(?:\.aspx)?$/i.test(new URL(listUrl).pathname)) {
            const hasSearch = await page
                .locator("#MainContent_btnSearch")
                .count()
                .catch(() => 0);
            if (hasSearch) {
                await page
                    .locator("#MainContent_cbImages")
                    .check()
                    .catch(() => {});
                await page
                    .locator("#MainContent_cbUserImages")
                    .check()
                    .catch(() => {});
                await page
                    .locator("#MainContent_ddPageSizeCatalogue")
                    .selectOption("1000")
                    .catch(() => {});
                await page.locator("#MainContent_btnSearch").click();
                await page
                    .waitForLoadState("load", { timeout: 90000 })
                    .catch(() => {});
                await sleep(opts.delayMs);
            }
        }
        const links = await page.evaluate(() => {
            const out = [];
            for (const a of document.querySelectorAll(
                'a[href*="Meteorite.aspx?id="]',
            )) {
                const href = a.getAttribute("href");
                if (!href) continue;
                out.push(new URL(href, location.href).href);
            }
            return [...new Set(out)];
        });
        return links;
    }

    async function extractPaginationCandidates() {
        return page.evaluate(() => {
            const out = [];
            for (const a of document.querySelectorAll("a[href]")) {
                const href = a.getAttribute("href");
                if (!href) continue;
                const abs = new URL(href, location.href).href;
                if (/Meteorites\.aspx/i.test(abs) || /Default\.aspx/i.test(abs))
                    out.push(abs);
            }
            return [...new Set(out)];
        });
    }

    async function extractMeteoritePage(url) {
        await politeGoto(url);
        const data = await page.evaluate(() => {
            const text = (el) =>
                el ? el.textContent.trim().replace(/\s+/g, " ") : "";
            const title = document.title || "";
            const h1 =
                text(document.querySelector("h1")) ||
                text(document.querySelector("h2")) ||
                "";
            const allText = document.body ? document.body.innerText : "";
            const links = Array.from(document.querySelectorAll("a[href]")).map(
                (a) => {
                    const href = new URL(a.getAttribute("href"), location.href)
                        .href;
                    return { text: text(a), href };
                },
            );
            const images = Array.from(document.querySelectorAll("img"))
                .map((img) => ({
                    src: img.currentSrc || img.src || "",
                    alt: img.alt || "",
                    width: img.naturalWidth || img.width || 0,
                    height: img.naturalHeight || img.height || 0,
                }))
                .filter((img) => img.src);
            return { title, h1, allText, links, images };
        });

        const imageUrls = uniq(
            data.images.filter(looksLikeContentImage).map((img) => img.src),
        );

        const collectionLinks = uniq(
            data.links
                .filter((l) => /Collection\.aspx\?id=/i.test(l.href))
                .map((l) => ({ text: l.text, href: l.href })),
        );

        const meteoriteLink = uniq(
            data.links
                .filter((l) => /Meteorite\.aspx\?id=/i.test(l.href))
                .map((l) => ({ text: l.text, href: l.href })),
        );

        return {
            url,
            title: data.title,
            heading: data.h1,
            text: data.allText.slice(0, 8000),
            imageUrls,
            collectionLinks,
            relatedLinks: meteoriteLink,
        };
    }

    async function downloadImage(imgUrl, refUrl, label) {
        if (visitedImages.has(imgUrl)) return null;
        const res = await context.request.get(imgUrl);
        if (!res.ok()) return null;
        const body = await res.body();
        const urlObj = new URL(imgUrl);
        const baseName = safeName(
            path.basename(urlObj.pathname) || label || "image",
        );
        const ext =
            path.extname(baseName) || path.extname(urlObj.pathname) || ".jpg";
        const stem = path.basename(baseName, ext || undefined);
        const fileName = `${stem}${ext || ".jpg"}`;
        const filePath = path.join(imagesDir, fileName);
        await fs.writeFile(filePath, body);
        visitedImages.add(imgUrl);
        state.imageUrls = [...visitedImages];
        await saveJson(stateFile, {
            visitedPages: [...visitedPages],
            visitedMeteorites: [...visitedMeteorites],
            imageUrls: [...visitedImages],
        });
        return { imgUrl, filePath, refUrl };
    }

    const seedPages = uniq([
        new URL("Default.aspx", opts.baseUrl).href,
        new URL("Meteorites.aspx", opts.baseUrl).href,
    ]);
    const pageQueue = [...seedPages];
    const meteoriteQueue = [];

    const seedPageSet = new Set(seedPages);

    while (pageQueue.length) {
        const current = pageQueue.shift();
        if (visitedPages.has(current) && !seedPageSet.has(current)) continue;
        visitedPages.add(current);
        state.visitedPages = [...visitedPages];
        await saveJson(stateFile, {
            visitedPages: [...visitedPages],
            visitedMeteorites: [...visitedMeteorites],
            imageUrls: [...visitedImages],
        });

        console.log(`LIST ${current}`);
        let links = [];
        try {
            links = await extractMeteoriteLinks(current);
        } catch (err) {
            console.error(`Failed list page ${current}: ${err.message}`);
            continue;
        }

        for (const link of links) {
            if (!visitedMeteorites.has(link)) meteoriteQueue.push(link);
        }

        const extraPages = await extractPaginationCandidates().catch(() => []);
        for (const nextPage of extraPages) {
            if (!visitedPages.has(nextPage)) pageQueue.push(nextPage);
        }

        if (visitedPages.size >= opts.maxPages) break;
    }

    while (meteoriteQueue.length && records.length < opts.maxItems) {
        const current = meteoriteQueue.shift();
        if (visitedMeteorites.has(current)) continue;
        visitedMeteorites.add(current);
        state.visitedMeteorites = [...visitedMeteorites];
        await saveJson(stateFile, {
            visitedPages: [...visitedPages],
            visitedMeteorites: [...visitedMeteorites],
            imageUrls: [...visitedImages],
        });

        console.log(`ITEM ${current}`);
        let rec;
        try {
            rec = await extractMeteoritePage(current);
        } catch (err) {
            console.error(`Failed item ${current}: ${err.message}`);
            continue;
        }

        const itemIdMatch = current.match(/id=(\d+)/i);
        const itemId = itemIdMatch
            ? itemIdMatch[1]
            : safeName(rec.heading || rec.title);
        const itemDir = path.join(pagesDir, safeName(itemId));
        await ensureDir(itemDir);

        const imageDownloads = [];
        for (const imgUrl of rec.imageUrls) {
            const downloaded = await downloadImage(imgUrl, current, itemId);
            if (downloaded) imageDownloads.push(downloaded);
        }

        const payload = {
            url: current,
            title: rec.title,
            heading: rec.heading,
            collectionLinks: rec.collectionLinks,
            relatedLinks: rec.relatedLinks,
            imageUrls: rec.imageUrls,
            downloadedImages: imageDownloads,
            sourceText: rec.text,
        };
        await saveJson(path.join(itemDir, "page.json"), payload);
        records.push(payload);

        if (records.length >= opts.maxItems) break;
    }

    await saveJson(manifestFile, {
        baseUrl: opts.baseUrl,
        generatedAt: new Date().toISOString(),
        visitedPages: [...visitedPages],
        visitedMeteorites: [...visitedMeteorites],
        imageCount: visitedImages.size,
        itemCount: records.length,
        records,
    });

    await browser.close();
    console.log(
        `Done. Wrote ${records.length} items and ${visitedImages.size} images to ${outDir}`,
    );
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});
