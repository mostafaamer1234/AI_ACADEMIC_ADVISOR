# majors_graph.py
from fuzzywuzzy import process, fuzz
import io
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend


class MajorGraph:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.G = nx.DiGraph()
        # Store all major names
        self.major_names = self.df['Major Name'].tolist()
        self.course_types = {
            "Prescribed Courses": "Prescribed",
            "Additional Courses": "Additional",
            "General Education Requirements": "General Education",
            "Supporting Courses and Related Areas": "Supporting",
            "Special Requirements": "Special",
            "General Option Courses": "General Option",
            "Data Science Option Courses": "Data Science Option",
            "Applied Data Sciences Option Summary": "Applied Data Sciences Option",
            "Computational Data Sciences Option Summary": "Computational Data Sciences Option",
            "Statistical Modeling Data Sciences Option Summary": "Statistical Modeling Data Sciences Option",
            "Traditions of Innovation Option Summary": "Traditions of Innovation Option",
            "Writing and Literature in Context Option Summary": "Writing and Literature in Context Option",
            "Accounting Option Summary": "Accounting Option",
            "Business Analytics Option Summary": "Business Analytics Option",
            "Entrepreneurship Option Summary": "Entrepreneurship Option",
            "Financial Services Option Summary": "Financial Services Option",
            "Health Services Option Summary": "Health Services Option",
            "Individualized Business Option Summary": "Individualized Business Option",
            "Management and Marketing Option Summary": "Management and Marketing Option",
            "Ecology Option Summary": "Ecology Option",
            "General Biology Option Summary": "General Biology Option",
            "Genetics & Dev Bio Option Summary": "Genetics & Dev Bio Option",
            "Neuroscience Option Summary": "Neuroscience Option",
            "Plant Biology Option Summary": "Plant Biology Option",
            "Vertebrate Physiology Option Summary": "Vertebrate Physiology Option",
            "Application Development Option Summary": "Application Development Option",
            "Business Applications Option Summary": "Business Applications Option",
            "Cybersecurity Option Summary": "Cybersecurity Option",
            "Networking Option Summary": "Networking Option",
            "Security and Risk Analysis Option Summary": "Security and Risk Analysis Option",
            "Specialized Technology Option Summary": "Specialized Technology Option",
            "Commercial Recreation and Tourism Management Option Summary": "Commercial Recreation Management Option",
            "Community Recreation Management Option Summary": "Community Recreation Option",
            "Outdoor Recreation Management Option Summary": "Outdoor Recreation Option",
            "Professional Golf Management Option Summary": "Professional Golf Option"
        }

        self._build_graph()

    def _extract_courses(self, text):
        if pd.isna(text) or not isinstance(text, str):
            return []
        return [c.strip() for c in text.split(",") if c.strip()]

    def _build_graph(self):
        for _, row in self.df.iterrows():
            major = row["Major Name"]
            self.G.add_node(major, type="major")

            for col, rel in self.course_types.items():
                if col in row:
                    for course in self._extract_courses(row[col]):
                        self.G.add_node(course, type="course")
                        self.G.add_edge(major, course, relation=rel)

    def _find_matching_major(self, query):
        """Find the best matching major name using fuzzy matching."""
        # First try exact match
        exact_matches = [
            name for name in self.major_names if query.lower() in name.lower()]
        if exact_matches:
            return exact_matches[0]

        # If no exact match, use fuzzy matching
        best_match = process.extractOne(
            query, self.major_names, scorer=fuzz.token_sort_ratio)
        # Only return if similarity is at least 60%
        if best_match and best_match[1] >= 60:
            return best_match[0]
        return None

    def _get_major_options(self, major_name):
        """Get all available options for a major."""
        options = []
        for col in self.course_types:
            if "Option" in col and col in self.df.columns:
                option_name = col.replace(
                    " Option Summary", "").replace(" Option Courses", "")
                if not pd.isna(self.df.loc[self.df['Major Name'] == major_name, col].iloc[0]):
                    options.append(option_name)
        return options

    def draw_major_graph(self, major_name):
        # Find the best matching major name
        actual_major_name = self._find_matching_major(major_name)
        if not actual_major_name:
            raise ValueError(
                f"Could not find a matching major for '{major_name}'. Please try a different name.")

        # Get all options for this major
        options = self._get_major_options(actual_major_name)

        if not options:
            # If no options, create a single graph for the major
            return self._create_single_graph(actual_major_name)
        else:
            # Create a graph for each option
            return self._create_option_graphs(actual_major_name, options)

    def _create_single_graph(self, major_name):
        """Create a graph for a major without options."""
        sub_nodes = {major_name}
        for neighbor in self.G.neighbors(major_name):
            sub_nodes.add(neighbor)

        subG = self.G.subgraph(sub_nodes)
        pos = nx.spring_layout(subG, seed=42)
        edge_labels = nx.get_edge_attributes(subG, 'relation')

        plt.figure(figsize=(14, 10))
        nx.draw(subG, pos, with_labels=True, node_size=2000,
                node_color='lightblue', font_size=10, font_weight='bold')
        nx.draw_networkx_edge_labels(
            subG, pos, edge_labels=edge_labels, font_color='red')
        plt.title(f"Course Graph for {major_name}")

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return buf

    def _visualize_graph(self, G, ax=None, title=None):
        print(
            f"[DEBUG] Visualizing graph with {len(G.nodes())} nodes and {len(G.edges())} edges.")
        print(f"[DEBUG] Nodes: {list(G.nodes())}")
        print(f"[DEBUG] Edges: {list(G.edges(data=True))}")
        if len(G.nodes()) < 2 or len(G.edges()) < 1:
            print(
                "[WARNING] Graph too small to visualize (need at least 2 nodes and 1 edge). Skipping.")
            if title:
                if ax:
                    ax.set_title(title + " (Graph too small)", fontsize=14)
                else:
                    plt.title(title + " (Graph too small)", fontsize=14)
            if ax:
                ax.axis('off')
            else:
                plt.axis('off')
            return
        try:
            pos = graphviz_layout(G, prog="dot", args='-Grankdir=TB')
        except Exception as e:
            print(f"[ERROR] graphviz_layout failed: {e}")
            raise RuntimeError(f"Graphviz layout failed: {e}")
        # Node colors and shapes
        node_colors = ["lightgreen" if "or" in str(
            node).lower() else "lightblue" for node in G.nodes()]
        node_shapes = {node: "d" if "or" in str(
            node).lower() else "o" for node in G.nodes()}

        # Draw regular nodes (circles)
        regular_nodes = [
            node for node in G.nodes() if node_shapes[node] == "o"]
        nx.draw_networkx_nodes(G, pos, nodelist=regular_nodes, node_size=3000, node_color="lightblue",
                               edgecolors="black", node_shape="o", ax=ax)
        # Draw OR nodes (diamonds)
        or_nodes = [node for node in G.nodes() if node_shapes[node] == "d"]
        if or_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=or_nodes, node_size=2000, node_color="lightgreen",
                                   edgecolors="black", node_shape="d", ax=ax)

        # Edge types
        prereq_edges = [(u, v) for u, v, d in G.edges(
            data=True) if d.get("type", "prereq") == "prereq"]
        concurrent_edges = [(u, v) for u, v, d in G.edges(
            data=True) if d.get("type") == "concurrent"]

        nx.draw_networkx_edges(G, pos, edgelist=prereq_edges, arrowstyle="-|>", arrowsize=15,
                               edge_color="black", connectionstyle="arc3,rad=0.2", ax=ax)
        if concurrent_edges:
            nx.draw_networkx_edges(G, pos, edgelist=concurrent_edges, style="dashed", edge_color="red",
                                   arrowstyle="-|>", arrowsize=15, connectionstyle="arc3,rad=0.2", ax=ax)

        # Labels
        node_labels = {node: G.nodes[node].get(
            "label", node) for node in G.nodes()}
        nx.draw_networkx_labels(
            G, pos, labels=node_labels, font_size=10, font_weight="bold", ax=ax)

        if title:
            if ax:
                ax.set_title(title, fontsize=14)
            else:
                plt.title(title, fontsize=14)
        if ax:
            ax.axis('off')
        else:
            plt.axis('off')

    def _create_option_graphs(self, major_name, options):
        """Create separate graphs for each option of a major, with multi-row blocks for each category for readability."""
        def chunk_list(lst, n):
            # Split list into chunks of size n
            return [lst[i:i + n] for i in range(0, len(lst), n)]

        def custom_layout(major, prescribed, required, general, max_per_row=8):
            pos = {}
            y = 2
            pos[major] = (0, y)
            y -= 1
            # Prescribed block
            prescribed_chunks = chunk_list(prescribed, max_per_row)
            for row, chunk in enumerate(prescribed_chunks):
                for i, node in enumerate(chunk):
                    pos[node] = (i - (len(chunk)-1)/2, y - row)
            y -= len(prescribed_chunks)
            # Required block
            required_chunks = chunk_list(required, max_per_row)
            for row, chunk in enumerate(required_chunks):
                for i, node in enumerate(chunk):
                    pos[node] = (i - (len(chunk)-1)/2, y - row)
            y -= len(required_chunks)
            # General block
            general_chunks = chunk_list(general, max_per_row)
            for row, chunk in enumerate(general_chunks):
                for i, node in enumerate(chunk):
                    pos[node] = (i - (len(chunk)-1)/2, y - row)
            return pos

        num_options = len(options)
        fig, axes = plt.subplots(num_options, 1, figsize=(16, 6 * num_options))
        if num_options == 1:
            axes = [axes]

        for i, option in enumerate(options):
            G = nx.DiGraph()
            # Major node
            G.add_node(major_name)
            # Extract course categories
            option_col = f"{option} Option Summary" if f"{option} Option Summary" in self.df.columns else f"{option} Option Courses"
            prescribed = self._extract_courses(
                self.df.loc[self.df['Major Name'] == major_name, 'Prescribed Courses'].iloc[0]) if 'Prescribed Courses' in self.df.columns else []
            # Only use Additional Courses for blue row
            additional = []
            if 'Additional Courses' in self.df.columns:
                additional += self._extract_courses(
                    self.df.loc[self.df['Major Name'] == major_name, 'Additional Courses'].iloc[0])
            general = self._extract_courses(self.df.loc[self.df['Major Name'] == major_name,
                                            'General Education Requirements'].iloc[0]) if 'General Education Requirements' in self.df.columns else []

            # Helper to wrap long labels
            def wrap_label(text, max_len=18):
                if len(text) <= max_len:
                    return text
                # Try to break at the nearest space to the middle
                mid = len(text) // 2
                before = text.rfind(' ', 0, mid)
                after = text.find(' ', mid)
                if before == -1 and after == -1:
                    return text  # no spaces, can't wrap
                if before == -1:
                    split = after
                elif after == -1:
                    split = before
                else:
                    split = before if (
                        mid - before) <= (after - mid) else after
                return text[:split] + '\n' + text[split+1:]

            # Add nodes and edges
            for node in prescribed:
                G.add_node(node)
                G.add_edge(major_name, node, style='solid', color='black')
            for node in additional:
                G.add_node(node)
                G.add_edge(major_name, node, style='dashed', color='black')
            for node in general:
                G.add_node(node)
                G.add_edge(major_name, node, style='solid', color='red')
            # Custom layout with multi-row blocks
            pos = custom_layout(major_name, prescribed,
                                additional, general, max_per_row=8)
            # Prepare wrapped labels
            all_nodes = [major_name] + prescribed + additional + general
            node_labels = {node: wrap_label(str(node)) for node in all_nodes}
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, nodelist=[
                                   major_name], node_color='lightblue', node_size=4000, ax=axes[i])
            nx.draw_networkx_nodes(
                G, pos, nodelist=prescribed, node_color='lightgreen', node_size=3000, ax=axes[i])
            nx.draw_networkx_nodes(
                G, pos, nodelist=additional, node_color='lightblue', node_size=3000, ax=axes[i])
            nx.draw_networkx_nodes(
                G, pos, nodelist=general, node_color='lightcoral', node_size=3000, ax=axes[i])
            # Draw edges
            for u, v, d in G.edges(data=True):
                style = d.get('style', 'solid')
                color = d.get('color', 'black')
                nx.draw_networkx_edges(
                    G, pos, edgelist=[(u, v)], style=style, edge_color=color, width=2, ax=axes[i])
            # Draw wrapped labels
            nx.draw_networkx_labels(
                G, pos, labels=node_labels, font_size=7, font_weight='bold', ax=axes[i])
            axes[i].set_title(
                f"{major_name} - {option} Option: Prescribed (green), Additional (blue), General (red)")
            axes[i].axis('off')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return buf

    def print_hierarchy(self, node, level=0, visited=None):
        if visited is None:
            visited = set()
        if node in visited:
            return
        visited.add(node)

        indent = "  " * level
        print(f"{indent}{node}")

        relation_groups = {}
        for succ in self.G.successors(node):
            rel = self.G.edges[node, succ].get("relation", "Other")
            relation_groups.setdefault(rel, []).append(succ)

        for relation, courses in sorted(relation_groups.items()):
            print(f"{indent}  [{relation}]")
            for i, course in enumerate(sorted(courses)):
                branch = "├── " if i < len(courses) - 1 else "└── "
                print(f"{indent}    {branch}{course}")


mg = MajorGraph(
    "/Users/mostafa/newFianlLLMRepo/IAAS/LLMAcadamic-Advisor/LLM/all_abington_majors_combined(5).csv")

if __name__ == "__main__":
    # Create instance of MajorGraph
    mg = MajorGraph(
        "/Users/mostafa/newFianlLLMRepo/IAAS/LLMAcadamic-Advisor/LLM/all_abington_majors_combined(5).csv")
    # Print and draw the hierarchy for a specific major
    mg.print_hierarchy("Computer Science, B.S. (Abington)")
    mg.draw_major_graph("Computer Science, B.S. (Abington)")
