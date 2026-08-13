"""Tests for HtmlParser."""

from smartchunk.parsers.base import get_parser
from smartchunk.parsers.html_parser import HtmlParser


def test_html_parser_factory_detection():
    parser = get_parser("page.html")
    assert isinstance(parser, HtmlParser)


def test_html_parsing_structure():
    html_content = """
    <html>
    <body>
        <h1>Financial Report</h1>
        <p>The company performed well in Q3.</p>
        <h2>Revenue Table</h2>
        <table>
            <caption>Q3 Performance</caption>
            <tr><th>Product</th><th>Revenue</th></tr>
            <tr><td>Widget A</td><td>$10M</td></tr>
        </table>
        <img src="chart.png" alt="Revenue Chart">
    </body>
    </html>
    """
    parser = HtmlParser()
    sections = parser.parse_text(html_content, source="test.html")

    assert len(sections) >= 3

    # Check paragraph
    p_sec = [s for s in sections if "performed well" in s.text][0]
    assert p_sec.heading == "Financial Report"

    # Check table
    table_sec = [s for s in sections if s.content_type == "table"][0]
    assert table_sec.table is not None
    assert table_sec.table.headers == ["Product", "Revenue"]
    assert table_sec.table.rows == [["Widget A", "$10M"]]

    # Check figure
    fig_sec = [s for s in sections if s.content_type == "figure"][0]
    assert len(fig_sec.figures) == 1
    assert fig_sec.figures[0].caption == "Revenue Chart"
