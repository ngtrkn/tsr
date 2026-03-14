#!/usr/bin/env python3
"""
Command-line script to parse PubTables1M dataset
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from tsr.data.pub1m_parser import Pub1MParser, parse_pub1m_directory


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Parse PubTables1M XML and words JSON to model format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Parse single file
  python parse_pub1m.py --xml /path/to/file.xml --words /path/to/file_words.json --output output.json
  
  # Parse directory
  python parse_pub1m.py --xml /path/to/xml_dir --words /path/to/words_dir --output /path/to/output_dir --batch
  
  # With image directory
  python parse_pub1m.py --xml /path/to/xml_dir --words /path/to/words_dir --output /path/to/output_dir --image /path/to/images --batch
  
  # With visualization and HTML export
  python parse_pub1m.py --xml /path/to/file.xml --words /path/to/file_words.json --output output.json --visualize --html
        """
    )
    
    parser.add_argument(
        "--xml",
        type=str,
        required=True,
        help="Path to XML annotation file or directory"
    )
    parser.add_argument(
        "--words",
        type=str,
        required=True,
        help="Path to words JSON file or directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON file or directory"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to image file or directory (optional)"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process directory in batch mode (required if input is directory)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization image with bounding boxes drawn"
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Export table to HTML format"
    )
    
    args = parser.parse_args()
    
    xml_path = Path(args.xml)
    words_path = Path(args.words)
    output_path = Path(args.output)
    
    # Determine if batch processing
    is_batch = args.batch or xml_path.is_dir()
    
    if is_batch:
        if not xml_path.is_dir():
            print("Error: --xml must be a directory when using --batch")
            sys.exit(1)
        if not words_path.is_dir():
            print("Error: --words must be a directory when using --batch")
            sys.exit(1)
        
        print(f"Batch processing mode")
        print(f"  XML directory: {xml_path}")
        print(f"  Words directory: {words_path}")
        print(f"  Output directory: {output_path}")
        if args.image:
            print(f"  Image directory: {args.image}")
        
        parse_pub1m_directory(
            xml_dir=str(xml_path),
            words_dir=str(words_path),
            output_dir=str(output_path),
            image_dir=args.image,
            visualize=args.visualize,
            export_html=args.html
        )
    else:
        if not xml_path.is_file():
            print(f"Error: XML file not found: {xml_path}")
            sys.exit(1)
        if not words_path.is_file():
            print(f"Error: Words file not found: {words_path}")
            sys.exit(1)
        
        print(f"Processing single file")
        print(f"  XML: {xml_path}")
        print(f"  Words: {words_path}")
        print(f"  Output: {output_path}")
        
        parser_obj = Pub1MParser(
            xml_path=str(xml_path),
            words_path=str(words_path),
            image_path=args.image
        )
        parser_obj.save_json(str(output_path))
        
        # Generate visualization if requested
        if args.visualize:
            vis_output = output_path.with_suffix('.visualized.jpg')
            parser_obj.visualize_labels(str(vis_output))
        
        # Export HTML if requested
        if args.html:
            html_output = output_path.with_suffix('.html')
            parser_obj.export_html(str(html_output))


if __name__ == "__main__":
    main()

