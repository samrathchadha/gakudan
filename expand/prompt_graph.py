def save_to_json(self, filename: str = "./expand.json"):
    """
    Save the graph to a JSON file with explicit RAG connections.
    Uses orjson for better performance and reliability.
    Handles NumPy data types properly.

    Args:
        filename: Path to save the JSON file
    """
    temp_filename = f"{filename}.tmp"

    with self.lock:
        try:
            # Format the data explicitly for visualization
            data = {
                "directed": True,
                "multigraph": True,
                "graph": {},
                "nodes": [],
                "links": []
            }

            # Helper function to convert NumPy values to Python native types
            def convert_numpy(obj):
                import numpy as np
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj

            # Add nodes with all their attributes
            for node_id, attrs in self.graph.nodes(data=True):
                node_data = {"id": node_id}
                # Add all attributes, ensuring they're serializable
                for key, value in attrs.items():
                    try:
                        # Convert NumPy types to native Python types
                        value = convert_numpy(value)

                        # Handle large string attributes
                        if isinstance(value, str) and len(value) > 100000:
                            node_data[key] = value[:100000] + "... [truncated]"
                        else:
                            node_data[key] = value
                    except:
                        # Convert to string as fallback
                        try:
                            node_data[key] = str(value)
                        except:
                            logging.warning(f"Skipping unserializable attribute {key} for node {node_id}")

                data["nodes"].append(node_data)

            # Add all edges, explicitly marking edge types
            for u, v, attrs in self.graph.edges(data=True):
                link_data = {
                    "source": u,
                    "target": v
                }

                # Determine edge type and set appropriate flags
                edge_type = attrs.get("edge_type", "hierarchy")

                if edge_type == "rag":
                    link_data["rag_connection"] = True
                    link_data["edge_type"] = "rag"
                    link_data["link_type"] = "rag"  # For visualizer compatibility
                else:
                    link_data["edge_type"] = "hierarchy"
                    link_data["link_type"] = "hierarchy"  # For visualizer compatibility

                # Add all other attributes with NumPy conversion
                for key, value in attrs.items():
                    if key not in link_data:  # Don't overwrite
                        try:
                            # Convert NumPy types
                            value = convert_numpy(value)
                            link_data[key] = value
                        except:
                            try:
                                link_data[key] = str(value)
                            except:
                                logging.warning(f"Skipping unserializable attribute {key} for link {u}->{v}")

                data["links"].append(link_data)

            # Ensure directory exists
            os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)

            # Try using orjson's built-in NumPy support first
            try:
                # Use orjson for serialization with optimizations and NumPy support
                json_bytes = orjson.dumps(
                    data,
                    option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY
                )

                # Use atomic write pattern to prevent truncation
                with open(temp_filename, 'wb') as f:
                    f.write(json_bytes)
                    # Ensure data is written to disk
                    f.flush()
                    os.fsync(f.fileno())
            except TypeError as e:
                # Fall back to standard json if orjson still has issues
                logging.warning(f"orjson serialization failed: {e}. Falling back to standard json.")

                # Custom JSON encoder for NumPy types
                import json
                import numpy as np

                class NumpyEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if isinstance(obj, np.integer):
                            return int(obj)
                        elif isinstance(obj, np.floating):
                            return float(obj)
                        elif isinstance(obj, np.ndarray):
                            return obj.tolist()
                        return json.JSONEncoder.default(self, obj)

                # Use standard json with custom encoder
                with open(temp_filename, 'w') as f:
                    json.dump(data, f, indent=2, cls=NumpyEncoder)
                    f.flush()
                    os.fsync(f.fileno())

            # Atomically replace the target file
            os.replace(temp_filename, filename)

            # Log stats
            hierarchy_count = sum(1 for link in data["links"] if link.get("edge_type") == "hierarchy")
            rag_count = sum(1 for link in data["links"] if link.get("edge_type") == "rag")

            logging.info(f"Graph saved to {filename} with {len(data['nodes'])} nodes, "
                       f"{hierarchy_count} hierarchy connections, and {rag_count} RAG connections")
            logging.info(f"Saved file size: {os.path.getsize(filename) / 1024:.2f} KB")

            return True

        except Exception as e:
            logging.error(f"Error saving graph to {filename}: {e}", exc_info=True)

            # Clean up temp file if it exists
            if os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                except:
                    pass

            return False
