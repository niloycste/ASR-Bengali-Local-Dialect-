"""
CS-focused YouTube query profiles for Bengali dialect data collection.

The target is speech where young speakers naturally mix English while still
revealing their home dialect. Queries are grouped by bucket so the pipeline can
measure which content domains actually yield code-switching.
"""

from __future__ import annotations

import json


DIALECT_PROFILES: dict[str, dict[str, str]] = {
    "Old_Dhaka": {
        "dialect": "Old_Dhaka",
        "region": "পুরান ঢাকার",
        "speech": "ঢাকাইয়া কুট্টি ভাষায়",
        "roman": "puran dhaka dhakaiya kutti",
    },
    "Comilla": {
        "dialect": "Comilla",
        "region": "কুমিল্লার",
        "speech": "কুমিল্লার ভাষায়",
        "roman": "kumilla dialect",
    },
    "Chittagong": {
        "dialect": "Chittagong",
        "region": "চট্টগ্রামের",
        "speech": "চাটগাঁইয়া ভাষায়",
        "roman": "chatgaiya dialect",
    },
    "Sylhet": {
        "dialect": "Sylhet",
        "region": "সিলেটের",
        "speech": "সিলেটি ভাষায়",
        "roman": "sylheti dialect",
    },
    "Barishal": {
        "dialect": "Barishal",
        "region": "বরিশালের",
        "speech": "বরিশালের ভাষায়",
        "roman": "barishal dialect",
    },
    "Noakhali": {
        "dialect": "Noakhali",
        "region": "নোয়াখালীর",
        "speech": "নোয়াখালীর ভাষায়",
        "roman": "noakhali dialect",
    },
    "Rangpur": {
        "dialect": "Rangpur",
        "region": "রংপুরের",
        "speech": "রংপুরের ভাষায়",
        "roman": "rangpur dialect",
    },
    "Mymensingh": {
        "dialect": "Mymensingh",
        "region": "ময়মনসিংহের",
        "speech": "ময়মনসিংহের ভাষায়",
        "roman": "mymensingh dialect",
    },
    "Khulna": {
        "dialect": "Khulna",
        "region": "খুলনার",
        "speech": "খুলনার ভাষায়",
        "roman": "khulna dialect",
    },
    "Kushtia": {
        "dialect": "Kushtia",
        "region": "কুষ্টিয়ার",
        "speech": "কুষ্টিয়ার ভাষায়",
        "roman": "kushtia dialect",
    },
    "Tangail": {
        "dialect": "Tangail",
        "region": "টাঙ্গাইলের",
        "speech": "টাঙ্গাইলের ভাষায়",
        "roman": "tangail dialect",
    },
    "Kishoreganj": {
        "dialect": "Kishoreganj",
        "region": "কিশোরগঞ্জের",
        "speech": "কিশোরগঞ্জের ভাষায়",
        "roman": "kishoreganj dialect",
    },
    "Habiganj": {
        "dialect": "Habiganj",
        "region": "হবিগঞ্জের",
        "speech": "হবিগঞ্জের ভাষায়",
        "roman": "habiganj dialect",
    },
    "Narail": {
        "dialect": "Narail",
        "region": "নড়াইলের",
        "speech": "নড়াইলের ভাষায়",
        "roman": "narail dialect",
    },
    "Narsingdi": {
        "dialect": "Narsingdi",
        "region": "নরসিংদীর",
        "speech": "নরসিংদীর ভাষায়",
        "roman": "narsingdi dialect",
    },
    "Sandwip": {
        "dialect": "Sandwip",
        "region": "সন্দ্বীপের",
        "speech": "সন্দ্বীপের ভাষায়",
        "roman": "sandwip dialect",
    },
}


DIALECT_DOMAIN_TEMPLATES: dict[str, list[str]] = {
    "Creator_Vlog": [
        "{region} তরুণ youtuber vlog english mixed",
        "{speech} content creator vlog english mixed",
        "{region} podcast prank challenge english mixed",
        "{roman} youth vlog english bangla mixed",
    ],
    "Tech_Review": [
        "{region} phone review english bangla",
        "{speech} tech review ফোন রিভিউ",
        "{region} laptop unboxing gadget review english mixed",
        "{roman} gadget review english bangla mixed",
    ],
    "Campus_Career": [
        "{region} university student vlog english mixed",
        "{region} campus interview career job english",
        "{speech} internship job interview english mixed",
        "{roman} student vlog career english mixed",
    ],
    "Street_Interview": [
        "{region} street interview তরুণ english mixed",
        "{region} young people social media viral english",
        "{speech} public reaction english mixed",
        "{roman} interview youth english bangla mixed",
    ],
}


SPECIAL_BUCKETS: dict[str, dict[str, object]] = {
    "Sylhet_UK_CS": {
        "dialect": "Sylhet",
        "domain": "Diaspora_UK",
        "queries": [
            "british bangladeshi sylheti vlog english daily life",
            "sylheti english second generation uk vlog 2024",
            "british sylheti youth conversation english bangla",
            "uk bangladeshi sylheti dialect english mix",
            "sylheti comedian british vlog english",
            "tower hamlets bangladeshi sylheti community english",
            "british born bangladeshi sylheti english conversation",
            "sylheti reaction video english british",
        ],
    },
}


SHARED_CS_BUCKETS: dict[str, dict[str, object]] = {
    "Bangladesh__Tech_IT_CS": {
        "dialect": "unknown",
        "domain": "Tech_IT",
        "queries": [
            "bangladesh software engineer interview bangla english",
            "bangladeshi programmer vlog english mixed",
            "বাংলাদেশ tech reviewer phone review english bangla",
            "বাংলাদেশ AI data science podcast bangla english",
        ],
    },
    "Bangladesh__Campus_Career_CS": {
        "dialect": "unknown",
        "domain": "Campus_Career",
        "queries": [
            "bangladesh university student vlog english bangla",
            "বাংলাদেশ internship career job interview english mixed",
            "বাংলাদেশ study abroad student podcast bangla english",
            "বাংলাদেশ campus interview career social media english",
        ],
    },
    "Bangladesh__Gaming_Reaction_CS": {
        "dialect": "unknown",
        "domain": "Gaming_Reaction",
        "queries": [
            "bangladesh gaming live stream bangla english",
            "বাংলাদেশ free fire pubg stream english mixed",
            "bangladeshi reaction video english bangla",
            "বাংলাদেশ gamer vlog english mixed",
        ],
    },
    "Bangladesh__Social_Media_CS": {
        "dialect": "unknown",
        "domain": "Social_Media",
        "queries": [
            "bangladesh content creator podcast english bangla",
            "বাংলাদেশ social media influencer vlog english mixed",
            "বাংলাদেশ viral challenge prank english bangla",
            "bangladeshi youtube shorts creator english mixed",
        ],
    },
    "Bangladesh__Freelancing_Startup_CS": {
        "dialect": "unknown",
        "domain": "Freelancing_Startup",
        "queries": [
            "bangladesh freelancing fiverr client meeting bangla english",
            "বাংলাদেশ startup founder podcast english bangla",
            "বাংলাদেশ digital marketing freelancer vlog english mixed",
            "bangladeshi remote job career vlog english mixed",
        ],
    },
}


def _expand_queries(profile: dict[str, str], templates: list[str]) -> list[str]:
    """Fill query templates for a single dialect."""
    return [
        template.format(
            region=profile["region"],
            speech=profile["speech"],
            roman=profile["roman"],
        ).strip()
        for template in templates
    ]


def build_youtube_buckets() -> dict[str, dict[str, object]]:
    """Return bucket_name -> {dialect, domain, queries}."""
    buckets: dict[str, dict[str, object]] = {}

    for dialect_name, profile in DIALECT_PROFILES.items():
        for domain, templates in DIALECT_DOMAIN_TEMPLATES.items():
            buckets[f"{dialect_name}__{domain}"] = {
                "dialect": profile["dialect"],
                "domain": domain,
                "queries": _expand_queries(profile, templates),
            }

    buckets.update(SPECIAL_BUCKETS)
    buckets.update(SHARED_CS_BUCKETS)
    return buckets


YOUTUBE_BUCKETS = build_youtube_buckets()

YOUTUBE_FOLDER_METADATA = {
    name: {"dialect": spec["dialect"], "domain": spec["domain"]}
    for name, spec in YOUTUBE_BUCKETS.items()
}


def write_query_catalog(path: str, buckets: dict[str, dict[str, object]] | None = None) -> None:
    """Persist the search plan for later audit."""
    payload = buckets or YOUTUBE_BUCKETS
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
