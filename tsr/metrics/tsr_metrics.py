"""
Table Structure Recognition Metrics
Implements TEDS (Tree-Edit-Distance-based Similarity) and other standard metrics
"""
import re
from typing import List, Dict, Tuple, Optional
from xml.etree import ElementTree as ET
from xml.dom import minidom


def tokens_to_html(tokens: List[str], vocab: Optional[Dict[str, int]] = None) -> str:
    """
    Convert token sequence to HTML table structure.
    Robust to common prediction errors (duplicate tags, missing row markers, etc.)
    """
    from tsr.data.serialization import (
        TABLE_START, TABLE_END, ROW_START, ROW_END,
        CELL_START, CELL_END, HEADER_START, HEADER_END,
        XMIN_TOKEN, YMIN_TOKEN, XMAX_TOKEN, YMAX_TOKEN,
        SEP_TOKEN, EOS_TOKEN, BOS_TOKEN
    )

    CELL_OPENERS = {CELL_START, HEADER_START}
    CELL_CLOSERS = {CELL_END, HEADER_END}
    STOP_TOKENS = (CELL_CLOSERS | {SEP_TOKEN, EOS_TOKEN, TABLE_END,
                    ROW_START, ROW_END, TABLE_START} | CELL_OPENERS)

    html_parts = []
    i = 0
    in_table = False
    in_row = False

    def _is_bbox_token(t: str) -> bool:
        return (t.startswith(XMIN_TOKEN) or t.startswith(YMIN_TOKEN) or
                t.startswith(XMAX_TOKEN) or t.startswith(YMAX_TOKEN))

    while i < len(tokens):
        token = tokens[i]

        if token in (BOS_TOKEN, SEP_TOKEN):
            i += 1
            continue

        if token == TABLE_START:
            if not in_table:
                html_parts.append("<table>")
                in_table = True
            i += 1
            continue

        if token == TABLE_END or token == EOS_TOKEN:
            if in_row:
                html_parts.append("</tr>")
                in_row = False
            if in_table:
                html_parts.append("</table>")
                in_table = False
            break

        if token == ROW_START:
            if in_row:
                html_parts.append("</tr>")
            html_parts.append("<tr>")
            in_row = True
            i += 1
            continue

        if token == ROW_END:
            if in_row:
                html_parts.append("</tr>")
                in_row = False
            i += 1
            continue

        if token in CELL_OPENERS:
            if not in_table:
                html_parts.append("<table>")
                in_table = True
            if not in_row:
                html_parts.append("<tr>")
                in_row = True

            tag = "th" if token == HEADER_START else "td"

            has_bbox = (i + 4 < len(tokens) and
                        tokens[i+1].startswith(XMIN_TOKEN) and
                        tokens[i+2].startswith(YMIN_TOKEN) and
                        tokens[i+3].startswith(XMAX_TOKEN) and
                        tokens[i+4].startswith(YMAX_TOKEN))

            content_start = i + 5 if has_bbox else i + 1

            content_tokens = []
            content_end = content_start
            while content_end < len(tokens) and tokens[content_end] not in STOP_TOKENS:
                t = tokens[content_end]
                if not _is_bbox_token(t):
                    content_tokens.append(t)
                content_end += 1

            content = "".join(content_tokens)
            content = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

            html_parts.append(f"<{tag}>{content}</{tag}>")

            if content_end < len(tokens) and tokens[content_end] in CELL_CLOSERS:
                i = content_end + 1
            else:
                i = content_end
            continue

        if token in CELL_CLOSERS:
            i += 1
            continue

        i += 1

    if in_row:
        html_parts.append("</tr>")
    if in_table and (not html_parts or html_parts[-1] != "</table>"):
        html_parts.append("</table>")

    return "".join(html_parts)


def normalize_html(html: str) -> str:
    """Normalize HTML for comparison"""
    # Remove extra whitespace
    html = re.sub(r'\s+', ' ', html)
    html = html.strip()
    return html


def tree_edit_distance(tree1: ET.Element, tree2: ET.Element) -> int:
    """
    Calculate tree edit distance between two XML trees
    Includes both structure and content in comparison
    """
    def get_tree_representation(node: ET.Element, depth: int = 0) -> List[Tuple[str, str, int]]:
        """Get tree representation as list of (tag, text, depth) tuples"""
        # Get text content (including from child text nodes)
        text = ""
        if node.text:
            text += node.text.strip()
        # Also check tail text (text after element)
        if node.tail:
            text += " " + node.tail.strip()
        text = text.strip()
        
        representation = [(node.tag, text, depth)]
        for child in node:
            representation.extend(get_tree_representation(child, depth + 1))
        return representation
    
    try:
        repr1 = get_tree_representation(tree1)
        repr2 = get_tree_representation(tree2)
        
        # Edit distance on structure + content
        m, n = len(repr1), len(repr2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if repr1[i-1] == repr2[j-1]:
                    # Exact match (tag, text, depth)
                    dp[i][j] = dp[i-1][j-1]
                else:
                    # Check if tags match (structure similarity)
                    tag_match = (repr1[i-1][0] == repr2[j-1][0])
                    # If tags match but content differs, cost is 0.5
                    # If tags differ, cost is 1.0
                    if tag_match:
                        # Same structure, different content
                        cost = 0.5
                    else:
                        # Different structure
                        cost = 1.0
                    dp[i][j] = cost + min(
                        dp[i-1][j],      # Delete
                        dp[i][j-1],      # Insert
                        dp[i-1][j-1]     # Replace
                    )
        
        return int(dp[m][n] * 2)  # Scale to integer
    except Exception as e:
        # If parsing fails, return large distance
        return 1000


def _sanitize_html(html: str) -> str:
    """Fix common HTML issues that break XML parsing"""
    html = re.sub(r'(<table>)+', '<table>', html)
    html = re.sub(r'(</table>)+', '</table>', html)
    if not html.startswith('<table>'):
        html = '<table>' + html
    if not html.endswith('</table>'):
        html = html + '</table>'
    return html


def calculate_teds(pred_html: str, gt_html: str) -> float:
    """
    Calculate TEDS (Tree-Edit-Distance-based Similarity) score
    
    TEDS = 1 - (edit_distance / max(len(pred_tree), len(gt_tree)))
    
    Args:
        pred_html: Predicted HTML table string
        gt_html: Ground truth HTML table string
    Returns:
        TEDS score between 0 and 1 (higher is better)
    """
    try:
        pred_html = normalize_html(pred_html)
        gt_html = normalize_html(gt_html)

        def _try_parse(html: str) -> Optional[ET.Element]:
            for attempt in [html, _sanitize_html(html)]:
                try:
                    return ET.fromstring(f"<root>{attempt}</root>")
                except ET.ParseError:
                    continue
            return None

        pred_tree = _try_parse(pred_html)
        gt_tree = _try_parse(gt_html)

        if pred_tree is None or gt_tree is None:
            return 0.0
        
        edit_dist = tree_edit_distance(pred_tree, gt_tree)
        
        pred_size = len(list(pred_tree.iter()))
        gt_size = len(list(gt_tree.iter()))
        max_size = max(pred_size, gt_size, 1)
        
        teds = 1.0 - (edit_dist / max_size)
        return max(0.0, min(1.0, teds))
        
    except Exception:
        return 0.0


def calculate_table_metrics(
    pred_tokens: List[str],
    gt_tokens: List[str],
    vocab: Optional[Dict[str, int]] = None,
) -> Dict[str, float]:
    """
    Calculate comprehensive table structure recognition metrics
    
    Args:
        pred_tokens: Predicted token sequence
        gt_tokens: Ground truth token sequence
        vocab: Optional vocabulary (not needed if tokens are strings)
    Returns:
        Dictionary with metrics:
        - teds: TEDS score (0-1)
        - structure_f1: F1 score on structure tokens
        - content_f1: F1 score on content tokens
    """
    from tsr.data.serialization import (
        TABLE_START, TABLE_END, ROW_START, ROW_END,
        CELL_START, CELL_END, HEADER_START, HEADER_END
    )
    
    metrics = {}
    
    # Convert to HTML and calculate TEDS
    try:
        pred_html = tokens_to_html(pred_tokens, vocab)
        gt_html = tokens_to_html(gt_tokens, vocab)
        metrics['teds'] = calculate_teds(pred_html, gt_html)
    except Exception as e:
        metrics['teds'] = 0.0
    
    # Structure token F1
    structure_tokens = {TABLE_START, TABLE_END, ROW_START, ROW_END, 
                        CELL_START, CELL_END, HEADER_START, HEADER_END}
    
    pred_struct = [t for t in pred_tokens if t in structure_tokens]
    gt_struct = [t for t in gt_tokens if t in structure_tokens]
    
    # Calculate precision, recall, F1 for structure tokens
    if len(pred_struct) > 0 and len(gt_struct) > 0:
        # Simple token-level matching
        pred_set = set(pred_struct)
        gt_set = set(gt_struct)
        
        tp = len(pred_set & gt_set)
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics['structure_precision'] = precision
        metrics['structure_recall'] = recall
        metrics['structure_f1'] = f1
    else:
        metrics['structure_precision'] = 0.0
        metrics['structure_recall'] = 0.0
        metrics['structure_f1'] = 0.0
    
    return metrics

