# sunqh.top — Personal Site & Tech Blog

My personal website and technical blog, built with [Jekyll](https://jekyllrb.com) and shipped as a **Progressive Web App (PWA)** at **[sunqh.top](https://sunqh.top)**.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
![Jekyll](https://img.shields.io/badge/Jekyll-CC0000?logo=jekyll&logoColor=white)
![Bootstrap](https://img.shields.io/badge/Bootstrap-7952B3?logo=bootstrap&logoColor=white)

## Features

- Jekyll static site with archive, tag pages and offline support
- PWA: service worker (`sw.js`), offline page and client-side search (`search.json`)
- Responsive, Bootstrap-based layout

## Structure

```text
├── _posts/          # blog posts
├── _layouts/        # Jekyll layouts
├── _includes/       # partials
├── img/             # images & assets
├── pwa/             # PWA assets
└── sw.js            # service worker
```

## Development

```bash
bundle install
bundle exec jekyll serve
```

## License

[Apache License 2.0](LICENSE)
