# ابزار کرول و استخراج مستندات برای امبد

این اسکریپت کل صفحات یک راهنمای نرم‌افزار (Documentation) را کرول می‌کند، متن‌های تمیز را به همراه مسیر عنوان‌ها (Heading Path) و شاخص پاراگراف استخراج می‌کند و در قالب JSONL خروجی می‌دهد؛ مناسب برای امبد کردن توسط مدل‌های برداری.

## نصب

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

روی لینوکس/مک:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## اجرا

مثال:

```bash
python crawler.py --start-url "https://docs.example.com/guide/" --output "docs.paragraphs.jsonl" --combined-output "docs.combined.jsonl" --combine-max-chars 1200 --max-pages 1000 --delay 0.2
```

پارامترها:

- `--start-url`: آدرس ریشه‌ی مستندات (همان جایی که منو/راهنما شروع می‌شود).
- `--output`: مسیر خروجی JSONL برای رکوردهای «سطح پاراگراف».
- `--combined-output`: مسیر خروجی JSONL اختیاری که پاراگراف‌های متوالی در یک عنوان را تا سقف کاراکتر ادغام می‌کند.
- `--combine-max-chars`: حداکثر تعداد کاراکتر برای هر چانک ترکیبی (برای امبد بهینه‌تر).
- `--max-pages`: حداکثر تعداد صفحات برای کرول.
- `--delay`: تاخیر بین درخواست‌ها (ثانیه) برای رعایت ادب درخواست‌ها.
- `--user-agent`: مقدار User-Agent سفارشی (اختیاری).

## قالب خروجی

هر خط JSON شامل فیلدهای زیر است:

```jsonc
{
  "url": "https://docs.example.com/guide/install",
  "page_title": "Install - Example Docs",
  "heading_path": ["Getting Started", "Install"],
  "paragraph_index": 12,
  "text": "پاراگراف تمیز شده...",
  "title_with_path": "Install - Example Docs > Getting Started > Install"
}
```

- `heading_path`: مسیر عنوان‌ها از بالاترین سطح تا عنوان جاری.
- `paragraph_index`: شمارنده‌ی پاراگراف در صفحه.
- `text`: متن تمیز مناسب برای امبد.

## نکات

- اسکریپت `robots.txt` را در همان دامنه بررسی می‌کند و فقط صفحات HTML داخلی همان دامنه را دنبال می‌کند.
- لینک‌های خارجی، `mailto:`، `tel:`، `javascript:` و قطعه‌ها (fragment) حذف می‌شوند.
- متن‌ها از `p`, `li`, `pre`, `code`, `blockquote` و متن‌های آزاد در `section/article` جمع‌آوری می‌شوند.
- اگر صفحه‌ای عنوان `<title>` نداشته باشد، از اولین `h1` استفاده می‌شود.

## بهترین شیوه‌ها برای امبد

- اگر مدل امبد شما محدودیت توکن/کاراکتر دارد، از `--combined-output` به همراه `--combine-max-chars` استفاده کنید تا پاراگراف‌های مرتبط در یک عنوان با هم ادغام شوند.
- معمولا ادغام تا حدود 800–1500 کاراکتر کیفیت خوبی می‌دهد و فراخوانی‌ها را کمتر می‌کند.


