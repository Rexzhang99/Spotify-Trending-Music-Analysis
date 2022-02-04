## What kind of music is trending?

**Team member: Zhendong Zhang**

### Aim

1.   What are the main reasons makes specific music popular.
2.   Predict what kind of music is going to trend in the future. 

### Data Content

#### ARTISTS (Size: 49723)

- id: artist's unique identifier;
- real_name: artist's real name;
- art_name: the career name the artist has chosen for him/herself;
- role: the artist's main role (singer, guitarist, dancer, drummer... );
- year_of_birth: the artist's year of birth;
- email: the artist's email. It's not necessarily unique as it can be also a common email;
- country: where the artist comes from;
- city: the city in which the artist lives. It can be in a different country from what stated above;
- zip_code: the postal zip code.

#### ALBUMS (Size: 67815)

- id: the album identifier;
- artist_id: the artist identifier;
- album_title: the title of the album;
- genre: the genre of the album. An artist can release albums of different genres;
- year_of_pub: the year the album was published;
- num_of_tracks: how many tracks there are in the album (a small number can mean longer tracks);
- num_of_sales: how many sales the album has made in the first month after the release;
- rolling_stone_critic: how magazine Rolling Stone has rated the album;
- mtv_critic: how MTV has rated the album;
- music_maniac_critic: how review site Music Maniac has rated the album.

### Methods going to use

1.   Linear regression
2.   Partial least squares regression
3.   Ridge regression 
4.   Lasso regression
5.   Plastic net regression