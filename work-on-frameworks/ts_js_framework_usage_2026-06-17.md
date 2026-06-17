# TS/JS Framework Usage Among Users

Date: 2026-06-17

## Summary

This note ranks the most used TypeScript/JavaScript frameworks and adjacent tooling among external SonarCloud users, based on detected NPM dependencies in active projects.

Population:

- 359 external organizations
- 4,410 active projects with NPM dependencies

Key takeaway:

- If all TS/JS tooling is included, the most prevalent families are Babel, PostCSS, TypeScript, ESLint, Rollup, esbuild, Vite, and Webpack.
- If the scope is narrowed to application frameworks, Express and React are the clear leaders, followed by Angular and Vue.
- For testing, Jest leads, followed by Vitest and Playwright.

## Ranked Table

| family | category | orgs | projects | org_pct | project_pct |
| --- | --- | ---: | ---: | ---: | ---: |
| Babel | compiler/transpiler | 323 | 3660 | 90.0 | 83.0 |
| PostCSS | css tooling | 321 | 2827 | 89.4 | 64.1 |
| TypeScript | language/tooling | 312 | 3337 | 86.9 | 75.7 |
| ESLint | linting | 308 | 3270 | 85.8 | 74.1 |
| Rollup | build tool | 275 | 1961 | 76.6 | 44.5 |
| esbuild | build tool | 273 | 1941 | 76.0 | 44.0 |
| Vite | build tool | 260 | 1646 | 72.4 | 37.3 |
| Webpack | build tool | 251 | 1953 | 69.9 | 44.3 |
| Express | server framework | 250 | 2189 | 69.6 | 49.6 |
| React | ui framework/library | 244 | 1516 | 68.0 | 34.4 |
| Jest | test framework | 228 | 2125 | 63.5 | 48.2 |
| SWC | compiler/transpiler | 188 | 945 | 52.4 | 21.4 |
| Tailwind CSS | css framework | 178 | 642 | 49.6 | 14.6 |
| Vitest | test framework | 158 | 817 | 44.0 | 18.5 |
| Playwright | e2e test framework | 139 | 537 | 38.7 | 12.2 |
| Karma | test runner | 124 | 836 | 34.5 | 19.0 |
| Angular | ui framework | 122 | 887 | 34.0 | 20.1 |
| Vue | ui framework | 115 | 413 | 32.0 | 9.4 |
| Jasmine | test framework | 113 | 638 | 31.5 | 14.5 |
| Remix | meta-framework | 104 | 366 | 29.0 | 8.3 |
| Storybook | component/dev tooling | 89 | 433 | 24.8 | 9.8 |
| Cypress | e2e test framework | 80 | 249 | 22.3 | 5.6 |
| Mocha | test framework | 78 | 320 | 21.7 | 7.3 |
| Next.js | meta-framework | 75 | 268 | 20.9 | 6.1 |
| Preact | ui framework | 64 | 115 | 17.8 | 2.6 |
| Nuxt | meta-framework | 55 | 163 | 15.3 | 3.7 |
| NestJS | server framework | 53 | 267 | 14.8 | 6.1 |
| SolidJS | ui framework | 18 | 24 | 5.0 | 0.5 |
| Svelte | ui framework | 13 | 13 | 3.6 | 0.3 |
| Astro | meta-framework | 5 | 5 | 1.4 | 0.1 |

## By Interpretation

### Application frameworks only

| family | orgs | projects | org_pct |
| --- | ---: | ---: | ---: |
| Express | 250 | 2189 | 69.6 |
| React | 244 | 1516 | 68.0 |
| Angular | 122 | 887 | 34.0 |
| Vue | 115 | 413 | 32.0 |
| Remix | 104 | 366 | 29.0 |
| Next.js | 75 | 268 | 20.9 |
| Preact | 64 | 115 | 17.8 |
| Nuxt | 55 | 163 | 15.3 |
| NestJS | 53 | 267 | 14.8 |
| SolidJS | 18 | 24 | 5.0 |
| Svelte | 13 | 13 | 3.6 |
| Astro | 5 | 5 | 1.4 |

### Test frameworks and runners

| family | orgs | projects | org_pct |
| --- | ---: | ---: | ---: |
| Jest | 228 | 2125 | 63.5 |
| Vitest | 158 | 817 | 44.0 |
| Playwright | 139 | 537 | 38.7 |
| Karma | 124 | 836 | 34.5 |
| Jasmine | 113 | 638 | 31.5 |
| Cypress | 80 | 249 | 22.3 |
| Mocha | 78 | 320 | 21.7 |

## Method

Relevant assets were identified through Atlan, then queried through Sonardata.

Primary warehouse tables:

- `sonar.fct_sc_sca_release`
- `sonar.fct_sc_project`

Population and join:

- Filtered to `package_manager = 'NPM'`
- Filtered to external orgs with `is_internal = false`
- Filtered to active projects with `is_active_project = true`
- Joined `sonar.fct_sc_sca_release.component_uuid = sonar.fct_sc_project.project_uuid_v4`

Aggregation logic:

- Counted distinct organizations via `organization_uuid_v4`
- Counted distinct projects via `project_uuid_v4`
- Mapped package names into framework/tool families using a curated package-family mapping

## Caveats

- This is dependency-detection based, not package.json intent based.
- Results reflect detected NPM packages, so they include both frameworks and widely used tooling.
- Broad families like Babel and PostCSS score very high because they are common across many frontend stacks.
- Some families can be undercounted or overcounted depending on package naming conventions and monorepo structure.

## File

Raw export: [ts_js_framework_usage_2026-06-17.csv](/Users/romain.birling/ts_js_framework_usage_2026-06-17.csv)
