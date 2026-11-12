# Licensing

Copyright (C) 2026 Hi9841. Trit has two licences.

## 1. AGPL-3.0 (free)

Trit is free software under the GNU Affero General Public License,
version 3 only ([LICENSE](LICENSE)). Use it free of charge for:

- personal use,
- academic work and teaching,
- research.

The AGPL-3.0 has these conditions:

- When you give a copy of Trit, or of a program that contains Trit, give
  its complete source code under the AGPL-3.0.
- When users get access to a changed Trit over a network (for example, a
  web service or an API), give them the complete source code of that
  service under the AGPL-3.0.
- Keep the copyright notices and the licence text.

Trit stays open source. Anyone can read, change, and share the code
under the AGPL-3.0.

## 2. Companies: commercial rights (paid)

A company that uses Trit in a commercial product or a paid service must
get commercial rights from the author in one of these ways:

1. **Buy a commercial licence.** You pay a fee, and you do not have to
   publish your source code under the AGPL-3.0.
2. **Pay for a custom agreement**, for example support, a feature, or an
   exclusive term.
3. **Acquire or hire** (acqui-hire) the author and Trit, by agreement.

Refer to [COMMERCIAL.md](COMMERCIAL.md).

Examples of commercial use:

- a closed-source application, device, or SDK that contains Trit,
- a paid or hosted service (SaaS, an API) that runs Trit,
- internal tools of a company that you do not publish under the AGPL-3.0.

If you are not sure that your use is commercial, ask
([COMMERCIAL.md](COMMERCIAL.md)).

## Earlier versions

Versions 0.12.0 and earlier were released under the PolyForm
Noncommercial License 1.0.0. Those copies stay under that licence.

## Contributions

To keep the two licences possible, a contributor must agree that the
author can release the contribution under the AGPL-3.0 and under the
commercial licence.

## Models and dependencies

The Trit licence does not apply to the models. Each model has its own
licence, and that licence applies when you use or ship the model. A Trit
commercial licence does not give you rights to a model.

These models forbid commercial use or limit it:

| Model | Licence | Commercial use |
|---|---|---|
| `mms-tts-eng` (the VITS voice) | CC-BY-NC-4.0 | **Forbidden** |
| `canary-1b-q8_0` | CC-BY-NC-4.0 | **Forbidden** |
| `granite-speech-5.0-470m-turboctc-nc-q8_0` | CC-BY-NC-SA-4.0 | **Forbidden** |
| `moonshine-{tiny,base}-{ar,ja,ko,uk,vi,zh}-q8_0` (12 models) | Moonshine AI Community License | **Limited:** register with Moonshine AI; a paid licence is necessary above US$1M yearly revenue |
| `medasr-q8_0` | Health AI Developer Foundations Terms | **Limited:** prohibited uses, and clinical use has more conditions |

All other models permit commercial use, some with attribution or other
conditions. All Rust dependencies have permissive licences (MIT,
Apache-2.0, Zlib, ISC, BSD, Unlicense, Unicode-3.0). They are
compatible with the AGPL-3.0 and with closed-source use.

[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) has the full list: each
model of the catalog, each dependency, and the data that ships in Trit.
