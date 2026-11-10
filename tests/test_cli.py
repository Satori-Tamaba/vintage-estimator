import json
import sys

import pytest
from PIL import Image

from vintage_estimator.api import cli
from vintage_estimator.schemas import EstimationResult, RetrievedItem


def test_cli_json_and_rgb(tmp_path, monkeypatch, capsys):
    path = tmp_path / "gray.png"
    Image.new("L", (4, 3)).save(path)
    monkeypatch.setattr(sys, "argv", ["vintage-estimator", str(path), "--json", "--top-k", "7"])

    def estimate(image, top_k):
        assert image.shape == (3, 4, 3)
        assert top_k == 7
        return EstimationResult(
            item="ваза", style=None, price=None,
            similar_items=[RetrievedItem(item_id="1", score=0.9, price=0)],
        )

    monkeypatch.setattr(cli, "estimate", estimate)
    cli.main()
    result = json.loads(capsys.readouterr().out)
    assert result["item"] == "ваза"
    assert result["price"] is None
    assert result["similar_items"][0]["price"] == 0


@pytest.mark.parametrize("exists", [True, False])
def test_cli_reports_unreadable_image(tmp_path, monkeypatch, capsys, exists):
    path = tmp_path / "broken.png"
    if exists:
        path.write_bytes(b"broken")
    monkeypatch.setattr(sys, "argv", ["vintage-estimator", str(path)])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 1
    output = capsys.readouterr()
    assert not output.out
    assert "не удалось прочитать изображение" in output.err


@pytest.mark.parametrize("corrupt", [True, False])
def test_cli_reports_catalog_error(tmp_path, monkeypatch, capsys, corrupt):
    path = tmp_path / "photo.png"
    Image.new("RGB", (4, 4)).save(path)
    if corrupt:
        (tmp_path / "embeddings.npz").write_bytes(b"broken archive")
    monkeypatch.setattr(sys, "argv", [
        "vintage-estimator", str(path), "--catalog-dir", str(tmp_path),
    ])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 1
    assert "не удалось загрузить каталог" in capsys.readouterr().err


def test_cli_rejects_nonpositive_top_k(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["vintage-estimator", "photo.png", "--top-k", "0"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
