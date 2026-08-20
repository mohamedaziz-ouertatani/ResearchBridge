from researchbridge.ingestion.normalize import normalize_doi


def test_normalize_doi_strips_https_prefix() -> None:
    assert normalize_doi("https://doi.org/10.1145/3442188.3445922") == "10.1145/3442188.3445922"


def test_normalize_doi_strips_http_prefix() -> None:
    assert normalize_doi("http://doi.org/10.1145/3442188.3445922") == "10.1145/3442188.3445922"


def test_normalize_doi_strips_dx_doi_org() -> None:
    assert normalize_doi("https://dx.doi.org/10.1145/3442188.3445922") == "10.1145/3442188.3445922"


def test_normalize_doi_strips_doi_scheme() -> None:
    assert normalize_doi("doi:10.1145/3442188.3445922") == "10.1145/3442188.3445922"


def test_normalize_doi_lowercases() -> None:
    assert normalize_doi("10.1145/ABC.DEF") == "10.1145/abc.def"


def test_normalize_doi_bare_value_unchanged_besides_case() -> None:
    assert normalize_doi("10.1145/3442188.3445922") == "10.1145/3442188.3445922"


def test_normalize_doi_strips_whitespace() -> None:
    assert normalize_doi("  10.1145/3442188.3445922  ") == "10.1145/3442188.3445922"


def test_normalize_doi_none_stays_none() -> None:
    assert normalize_doi(None) is None


def test_normalize_doi_empty_string_becomes_none() -> None:
    assert normalize_doi("") is None
    assert normalize_doi("   ") is None
