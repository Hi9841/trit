//! Numbers as English words, as `num2words` 0.5.14 writes them (the
//! library misaki calls): cardinals with "and" after hundreds, ordinals,
//! years ("nineteen eighty-four"), and decimals digit by digit.

const LOW: [&str; 20] = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
];
const TENS: [&str; 10] = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
];
/// 10^3 to 10^36, short scale.
const SCALES: [&str; 12] = [
    "thousand",
    "million",
    "billion",
    "trillion",
    "quadrillion",
    "quintillion",
    "sextillion",
    "septillion",
    "octillion",
    "nonillion",
    "decillion",
    "undecillion",
];
const ORDINALS: [(&str, &str); 12] = [
    ("one", "first"),
    ("two", "second"),
    ("three", "third"),
    ("four", "fourth"),
    ("five", "fifth"),
    ("six", "sixth"),
    ("seven", "seventh"),
    ("eight", "eighth"),
    ("nine", "ninth"),
    ("ten", "tenth"),
    ("eleven", "eleventh"),
    ("twelve", "twelfth"),
];

pub fn cardinal(n: u128) -> String {
    if n < 20 {
        return LOW[n as usize].to_string();
    }
    if n < 100 {
        let (tens, units) = ((n / 10) as usize, (n % 10) as usize);
        return if units == 0 {
            TENS[tens].to_string()
        } else {
            format!("{}-{}", TENS[tens], LOW[units])
        };
    }
    if n < 1000 {
        let (hundreds, rest) = (n / 100, n % 100);
        let head = format!("{} hundred", LOW[hundreds as usize]);
        return if rest == 0 {
            head
        } else {
            format!("{head} and {}", cardinal(rest))
        };
    }
    let mut scale = 1000u128;
    let mut index = 0;
    while index + 1 < SCALES.len() && n / scale >= 1000 {
        scale *= 1000;
        index += 1;
    }
    let (q, r) = (n / scale, n % scale);
    let head = format!("{} {}", cardinal(q), SCALES[index]);
    match r {
        0 => head,
        r if r < 100 => format!("{head} and {}", cardinal(r)),
        r => format!("{head}, {}", cardinal(r)),
    }
}

pub fn ordinal(n: u128) -> String {
    let text = cardinal(n);
    let (head, last) = match text.rfind(' ') {
        Some(i) => (&text[..=i], &text[i + 1..]),
        None => ("", text.as_str()),
    };
    let (stem, word) = match last.rfind('-') {
        Some(i) => (&last[..=i], &last[i + 1..]),
        None => ("", last),
    };
    let word = match ORDINALS.iter().find(|(c, _)| *c == word) {
        Some((_, o)) => o.to_string(),
        None => {
            let base = word
                .strip_suffix('y')
                .map_or(word.to_string(), |w| format!("{w}ie"));
            format!("{base}th")
        }
    };
    format!("{head}{stem}{word}")
}

/// A year: "nineteen eighty-four", "twenty ten", "nineteen oh-five",
/// "two thousand and five".
pub fn year(n: u128) -> String {
    let (high, low) = (n / 100, n % 100);
    if high == 0 || (high % 10 == 0 && low < 10) || high >= 100 {
        return cardinal(n);
    }
    let low = match low {
        0 => "hundred".to_string(),
        l if l < 10 => format!("oh-{}", cardinal(l)),
        l => cardinal(l),
    };
    format!("{} {low}", cardinal(high))
}

/// `num2words(float(text))` for a decimal like "3.25": the whole part,
/// "point", then each digit. A whole value ("3.0") is a cardinal.
pub fn decimal(text: &str) -> Option<String> {
    let value: f64 = text.parse().ok()?;
    if !value.is_finite() {
        return None;
    }
    if value.fract() == 0.0 && value.abs() < 1e30 {
        let whole = value.abs() as u128;
        let sign = if value < 0.0 { "minus " } else { "" };
        return Some(format!("{sign}{}", cardinal(whole)));
    }
    // Python's repr of the float: the shortest digits that read back the same.
    let repr = format!("{value}");
    let (whole, digits) = repr.split_once('.')?;
    let negative = whole.starts_with('-');
    let whole: u128 = whole.trim_start_matches('-').parse().ok()?;
    let mut out = vec![
        format!(
            "{}{}",
            if negative { "minus " } else { "" },
            cardinal(whole)
        ),
        "point".to_string(),
    ];
    out.extend(
        digits
            .chars()
            .filter_map(|c| c.to_digit(10))
            .map(|d| cardinal(u128::from(d))),
    );
    Some(out.join(" "))
}

/// The words of `text` (lowercase runs), as misaki splits num2words' output.
pub fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_ascii_lowercase())
        .filter(|w| !w.is_empty())
        .map(str::to_string)
        .collect()
}
