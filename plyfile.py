import os
import struct

import numpy as np


_PLY_TO_DTYPE = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}

_DTYPE_TO_PLY = {
    ("i", 1): "char",
    ("u", 1): "uchar",
    ("i", 2): "short",
    ("u", 2): "ushort",
    ("i", 4): "int",
    ("u", 4): "uint",
    ("f", 4): "float",
    ("f", 8): "double",
}

_STRUCT_FORMAT = {
    "i1": "b",
    "u1": "B",
    "i2": "h",
    "u2": "H",
    "i4": "i",
    "u4": "I",
    "f4": "f",
    "f8": "d",
}


class PlyProperty:
    def __init__(self, name, dtype):
        self.name = name
        self.dtype = np.dtype(dtype)


class PlyListProperty:
    def __init__(self, name, count_type, item_type):
        self.name = name
        self.count_type = count_type
        self.item_type = item_type


class PlyElement:
    def __init__(self, name, data, properties):
        self.name = name
        self.data = data
        self.properties = properties

    @classmethod
    def describe(cls, data, name):
        if data.dtype.names is None:
            raise ValueError("PlyElement.describe expects a structured numpy array")

        properties = []
        for field_name in data.dtype.names:
            field_dtype = data.dtype.fields[field_name][0]
            if field_dtype.subdtype is None:
                properties.append(PlyProperty(field_name, field_dtype))
                continue
            base_dtype, shape = field_dtype.subdtype
            if len(shape) != 1:
                raise ValueError(f"PLY list field {field_name} must be one-dimensional")
            properties.append(PlyListProperty(field_name, "uchar", dtype_to_ply(base_dtype)))
        return cls(name, data, properties)

    def __getitem__(self, key):
        return self.data[key]

    def __contains__(self, key):
        return self.data.dtype.names is not None and key in self.data.dtype.names

    def __len__(self):
        return len(self.data)


class PlyData:
    def __init__(self, elements, text=True, byte_order="<", comments=None):
        self.elements = list(elements)
        self.text = text
        self.byte_order = byte_order
        self.comments = list(comments or [])

    def __getitem__(self, key):
        for element in self.elements:
            if element.name == key:
                return element
        raise KeyError(key)

    @classmethod
    def read(cls, path):
        with open(os.fspath(path), "rb") as f:
            fmt, elements, comments = parse_header(f)
            parsed = []
            for element in elements:
                if fmt == "ascii":
                    parsed.append(read_ascii_element(f, element))
                elif fmt == "binary_little_endian":
                    parsed.append(read_binary_element(f, element, "<"))
                elif fmt == "binary_big_endian":
                    parsed.append(read_binary_element(f, element, ">"))
                else:
                    raise ValueError(f"Unsupported PLY format: {fmt}")
        byte_order = ">"
        if fmt in {"ascii", "binary_little_endian"}:
            byte_order = "<"
        return cls(parsed, text=(fmt == "ascii"), byte_order=byte_order, comments=comments)

    def write(self, path):
        if self.text:
            with open(os.fspath(path), "w", encoding="ascii") as f:
                write_header(f, self.elements, self.comments, "ascii")
                for element in self.elements:
                    write_ascii_element(f, element)
            return

        byte_order = ">" if self.byte_order == ">" else "<"
        fmt = "binary_big_endian" if byte_order == ">" else "binary_little_endian"
        with open(os.fspath(path), "wb") as f:
            header_lines = []
            write_header(header_lines, self.elements, self.comments, fmt)
            f.write("".join(header_lines).encode("ascii"))
            for element in self.elements:
                write_binary_element(f, element, byte_order)


def write_header(out, elements, comments, fmt):
    def write_line(line):
        if hasattr(out, "write"):
            out.write(line)
        else:
            out.append(line)

    write_line("ply\n")
    write_line(f"format {fmt} 1.0\n")
    for comment in comments:
        clean_comment = str(comment).replace("\n", " ").replace("\r", " ")
        write_line(f"comment {clean_comment}\n")
    for element in elements:
        write_line(f"element {element.name} {len(element.data)}\n")
        for prop in element.properties:
            if isinstance(prop, PlyListProperty):
                write_line(f"property list {prop.count_type} {prop.item_type} {prop.name}\n")
            else:
                write_line(f"property {dtype_to_ply(prop.dtype)} {prop.name}\n")
    write_line("end_header\n")


def dtype_to_ply(dtype):
    dtype = np.dtype(dtype)
    key = (dtype.kind, dtype.itemsize)
    if key not in _DTYPE_TO_PLY:
        raise ValueError(f"Unsupported PLY dtype: {dtype}")
    return _DTYPE_TO_PLY[key]


def parse_header(f):
    first = f.readline().decode("ascii").strip()
    if first != "ply":
        raise ValueError("Not a PLY file")

    fmt = None
    elements = []
    current = None
    comments = []
    for raw_line in f:
        line = raw_line.decode("ascii").strip()
        if not line:
            continue
        if line.startswith("comment"):
            comments.append(line[len("comment") :].strip())
            continue
        if line == "end_header":
            break
        parts = line.split()
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            current = {"name": parts[1], "count": int(parts[2]), "properties": []}
            elements.append(current)
        elif parts[0] == "property" and current is not None:
            if parts[1] == "list":
                current["properties"].append(PlyListProperty(parts[4], parts[2], parts[3]))
            else:
                current["properties"].append(PlyProperty(parts[2], ply_to_dtype(parts[1])))
    if fmt is None:
        raise ValueError("PLY header is missing format")
    return fmt, elements, comments


def ply_to_dtype(ply_type):
    if ply_type not in _PLY_TO_DTYPE:
        raise ValueError(f"Unsupported PLY property type: {ply_type}")
    return np.dtype(_PLY_TO_DTYPE[ply_type])


def element_dtype(element):
    fields = []
    for prop in element["properties"]:
        if isinstance(prop, PlyListProperty):
            fields.append((prop.name, object))
        else:
            fields.append((prop.name, prop.dtype))
    return np.dtype(fields)


def read_ascii_element(f, element):
    dtype = element_dtype(element)
    data = np.empty(element["count"], dtype=dtype)
    for row in range(element["count"]):
        tokens = []
        while not tokens:
            raw = f.readline()
            if not raw:
                raise ValueError(f"Unexpected end of PLY while reading {element['name']}")
            tokens = raw.decode("ascii").strip().split()

        cursor = 0
        for prop in element["properties"]:
            if isinstance(prop, PlyListProperty):
                count = int(tokens[cursor])
                cursor += 1
                values = tokens[cursor : cursor + count]
                cursor += count
                data[prop.name][row] = np.asarray(values, dtype=ply_to_dtype(prop.item_type))
            else:
                data[prop.name][row] = tokens[cursor]
                cursor += 1
    return PlyElement(element["name"], data, element["properties"])


def read_binary_element(f, element, endian):
    if not any(isinstance(prop, PlyListProperty) for prop in element["properties"]):
        dtype = binary_element_dtype(element, endian)
        raw = np.fromfile(f, dtype=dtype, count=element["count"])
        return PlyElement(element["name"], raw.astype(element_dtype(element), copy=False), element["properties"])

    dtype = element_dtype(element)
    data = np.empty(element["count"], dtype=dtype)
    for row in range(element["count"]):
        for prop in element["properties"]:
            if isinstance(prop, PlyListProperty):
                count = int(read_binary_scalar(f, prop.count_type, endian))
                values = [read_binary_scalar(f, prop.item_type, endian) for _ in range(count)]
                data[prop.name][row] = np.asarray(values, dtype=ply_to_dtype(prop.item_type))
            else:
                data[prop.name][row] = read_binary_scalar(f, dtype_to_ply(prop.dtype), endian)
    return PlyElement(element["name"], data, element["properties"])


def binary_element_dtype(element, endian):
    fields = []
    for prop in element["properties"]:
        dtype = prop.dtype.newbyteorder(endian)
        fields.append((prop.name, dtype))
    return np.dtype(fields)


def read_binary_scalar(f, ply_type, endian):
    dtype = ply_to_dtype(ply_type)
    fmt = _STRUCT_FORMAT[dtype.str[1:]]
    size = dtype.itemsize
    raw = f.read(size)
    if len(raw) != size:
        raise ValueError("Unexpected end of PLY binary payload")
    return struct.unpack(endian + fmt, raw)[0]


def write_ascii_element(f, element):
    names = element.data.dtype.names
    for row in element.data:
        values = []
        for prop in element.properties:
            value = row[prop.name]
            if isinstance(prop, PlyListProperty):
                items = np.asarray(value).reshape(-1).tolist()
                values.append(str(len(items)))
                values.extend(format_scalar(item) for item in items)
            else:
                values.append(format_scalar(value))
        f.write(" ".join(values) + "\n")


def write_binary_element(f, element, endian):
    if not any(isinstance(prop, PlyListProperty) for prop in element.properties):
        fields = []
        for prop in element.properties:
            fields.append((prop.name, prop.dtype.newbyteorder(endian)))
        dtype = np.dtype(fields)
        data = np.empty(len(element.data), dtype=dtype)
        for prop in element.properties:
            data[prop.name] = np.asarray(element.data[prop.name], dtype=dtype.fields[prop.name][0])
        data.tofile(f)
        return

    for row in element.data:
        for prop in element.properties:
            value = row[prop.name]
            if isinstance(prop, PlyListProperty):
                items = np.asarray(value).reshape(-1)
                write_binary_scalar(f, len(items), prop.count_type, endian)
                for item in items:
                    write_binary_scalar(f, item, prop.item_type, endian)
            else:
                write_binary_scalar(f, value, dtype_to_ply(prop.dtype), endian)


def write_binary_scalar(f, value, ply_type, endian):
    dtype = ply_to_dtype(ply_type)
    fmt = _STRUCT_FORMAT[dtype.str[1:]]
    scalar = np.asarray(value, dtype=dtype).item()
    f.write(struct.pack(endian + fmt, scalar))


def format_scalar(value):
    scalar = np.asarray(value).item()
    if isinstance(scalar, float):
        return repr(float(scalar))
    return str(int(scalar)) if isinstance(scalar, np.integer) else str(scalar)
